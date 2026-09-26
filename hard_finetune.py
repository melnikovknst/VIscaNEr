#!/usr/bin/env python3
"""Hard-negative fine-tuning of an already trained DINOv3-B retrieval model.

All mining uses train crops only. Validation images never enter a training
batch or the hard-neighbor graph. The original checkpoint is kept intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

import dinov3_retrieval as base
import full_finetune as full


class HardIdentitySampler(Sampler[list[int]]):
    """Two mined confusable pairs plus random identities in each P x K batch."""

    def __init__(self, labels, neighbors, weights, *, seed, epoch, p=8, k=2, multiplier=1.5):
        self.by_label = {int(x): np.flatnonzero(np.asarray(labels) == x)
                         for x in np.unique(labels)}
        self.identities = np.asarray(sorted(self.by_label), dtype=np.int64)
        if len(self.identities) < p or p < 4:
            raise ValueError('Hard batches require at least eight train identities')
        self.neighbors = neighbors
        self.weights = np.asarray([weights.get(int(x), 1.) for x in self.identities], dtype=float)
        self.weights /= self.weights.sum()
        self.seed, self.epoch, self.p, self.k = seed, epoch, p, k
        self.count = math.ceil(len(labels) / (p * k) * multiplier)

    def __len__(self):
        return self.count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.count):
            selected = []
            # Every batch includes two pairs mined from actual train query ->
            # gallery confusions, while half the identities remain random.
            for _ in range(2):
                choices = [i for i, x in enumerate(self.identities) if int(x) not in selected]
                probs = self.weights[choices] / self.weights[choices].sum()
                anchor = int(self.identities[rng.choice(choices, p=probs)])
                selected.append(anchor)
                hard = [x for x in self.neighbors.get(anchor, ())
                        if x in self.by_label and x not in selected]
                if hard:
                    selected.append(int(rng.choice(hard[:6])))
                else:
                    remaining = [x for x in self.identities if int(x) not in selected]
                    selected.append(int(rng.choice(remaining)))
            remaining = [x for x in self.identities if int(x) not in selected]
            selected.extend(map(int, rng.choice(remaining, self.p - len(selected), replace=False)))
            batch = []
            for label in selected:
                examples = self.by_label[label]
                batch.extend(rng.choice(examples, self.k, replace=len(examples) < self.k).tolist())
            rng.shuffle(batch)
            yield batch


@torch.inference_mode()
def mine_neighbors(model, index, cfg, device, max_queries=3, topk=6):
    """Mine only from train queries; gallery images are catalog references."""
    train = index[index.split.eq('train')]
    sample = (train.sample(frac=1, random_state=cfg.seed)
                   .groupby('label_id', sort=True).head(max_queries))
    gallery = index[index.split.eq('gallery')]
    query_emb, query_labels, _, _ = base.embed_loader(
        model, base.create_eval_loader(sample, cfg), device, cfg.amp, 'mine-train')
    ref_emb, ref_labels, _, _ = base.embed_loader(
        model, base.create_eval_loader(gallery, cfg), device, cfg.amp, 'mine-gallery')
    q = torch.as_tensor(query_emb, dtype=torch.float32)
    r = torch.as_tensor(ref_emb, dtype=torch.float32)
    similarity = q @ r.T
    ref_labels = np.asarray(ref_labels, dtype=np.int64)
    query_labels = np.asarray(query_labels, dtype=np.int64)
    eligible = set(map(int, train.label_id.unique()))
    scores = {label: {} for label in eligible}
    hardness = {label: 1. for label in eligible}
    for row, label in zip(similarity, query_labels):
        label = int(label)
        own = row[torch.as_tensor(ref_labels == label)].max().item()
        ranking = torch.argsort(row, descending=True).tolist()
        negatives = [int(ref_labels[i]) for i in ranking
                     if int(ref_labels[i]) in eligible and int(ref_labels[i]) != label][:topk]
        for rank, other in enumerate(negatives):
            scores[label][other] = scores[label].get(other, 0.) + 1. / (rank + 1)
        if negatives and int(ref_labels[ranking[0]]) != label:
            hardness[label] += 1.
        elif negatives:
            best_negative = max(row[torch.as_tensor(ref_labels == negatives[0])]).item()
            if own - best_negative < 0.08:
                hardness[label] += 0.5
    neighbors = {label: [other for other, _ in sorted(values.items(),
                  key=lambda item: -item[1])[:topk]] for label, values in scores.items()}
    print(f'MINING | train queries={len(query_labels)} | identities={len(eligible)} '
          f'| weighted difficult={sum(v > 1 for v in hardness.values())}', flush=True)
    return neighbors, hardness


def loader_for_epoch(records, cfg, neighbors, weights, epoch):
    transform, _ = base.build_transforms(cfg.image_size)
    dataset = base.WineRetrievalDataset(records, cfg.crops_root, cfg.refs_root,
                                        transform, two_views=True)
    sampler = HardIdentitySampler(records.label_id.to_numpy(), neighbors, weights,
                                  seed=cfg.seed, epoch=epoch)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
                      pin_memory=True, persistent_workers=cfg.num_workers > 0)


def check_initial_checkpoint(payload, cfg, dataset_kind, n_classes):
    saved = payload.get('config', {})
    expected_slug = ('viscaner-bottle-classifier-data' if dataset_kind == 'bottles'
                     else 'viscaner-dinov3-data')
    if expected_slug not in str(saved.get('crops_metadata_path', '')):
        raise ValueError('Checkpoint belongs to a different crop dataset')
    keys = ('seed', 'unseen_identity_fraction', 'val_seen_per_identity',
            'include_low_confidence_in_train', 'image_size', 'embedding_dim',
            'projection_hidden_dim', 'ce_temperature')
    for key in keys:
        if saved.get(key) != getattr(cfg, key):
            raise ValueError(f'Checkpoint/index configuration mismatch: {key}')
    if payload['model_state_dict']['classifier.weight'].shape[0] != n_classes:
        raise ValueError('Checkpoint classifier does not match index identities')


def run(args):
    import yaml
    cfg = base.PipelineConfig(**yaml.safe_load(Path(args.config).read_text())).resolved()
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Kaggle GPU; local CPU/Mac training is disabled')
    base.seed_everything(cfg.seed)
    device = torch.device('cuda')
    index, summary = base.prepare_retrieval_index(cfg, validate_files=not args.skip_file_validation)
    index_hash = hashlib.sha256(Path(cfg.index_path).read_bytes()).hexdigest()
    train = index[index.split.eq('train')].reset_index(drop=True)
    if train.empty or index[index.split.eq('val_seen')].empty:
        raise ValueError('Train and val_seen splits must be nonempty')
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    out_dir = Path(cfg.models_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    original = Path(args.init_checkpoint)
    if not original.is_file():
        raise FileNotFoundError(original)
    if args.resume_checkpoint and not Path(args.resume_checkpoint).is_file():
        raise FileNotFoundError(args.resume_checkpoint)
    init_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    source = torch.load(original, map_location='cpu', weights_only=False)
    n_classes = int(index.label_id.max()) + 1
    check_initial_checkpoint(source, cfg, args.dataset_kind, n_classes)
    model = base.DINOv3RetrievalModel(
        base.load_local_dinov3_backbone(cfg.weights_path, cfg.image_size),
        n_classes, cfg.embedding_dim, cfg.projection_hidden_dim,
        cfg.ce_temperature)
    state = base.normalize_retrieval_checkpoint_state_dict(model, source['model_state_dict'])
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.set_backbone_trainable(-1)
    report = full.trainability(model)
    if any(x['total'] != x['trainable'] for x in report.values()):
        raise RuntimeError(f'Not fully trainable: {report}')
    print('TRAINABLE |', report, flush=True)
    cfg = replace(cfg, head_lr=args.head_lr, backbone_lr=args.backbone_lr)
    optimizer = base.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: full.lr_factor(epoch, args.epochs, args.warmup))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp, init_scale=1024.)
    history, start_epoch, best, bad = [], 0, -1., 0
    baseline = None
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        if state.get('format') != 'hard-finetune-v1' or state['index_sha256'] != index_hash \
                or state['init_sha256'] != init_sha or state['dataset_kind'] != args.dataset_kind:
            raise ValueError('Resume state does not match this run')
        model.load_state_dict(base.normalize_retrieval_checkpoint_state_dict(
            model, state['model_state_dict']), strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        full._restore_rng(state, device)
        history, start_epoch, best, bad = (state['history'], state['epoch'],
                                            state['best_monitor'], state['no_improvement'])
        baseline = state['baseline']
        full.atomic_save(state['best_payload'], out_dir / 'best.pt')
    else:
        baseline, _ = base.evaluate_splits(model, index, cfg, device,
                                            splits=('val_seen', 'val_unseen'))
        best = float(baseline['val_seen']['recall_at_1'])
        (run_dir / 'baseline_metrics.json').write_text(json.dumps(baseline, indent=2))
        full.atomic_save({'model_state_dict': model.state_dict(), 'config': asdict(cfg),
                          'epoch': 0, 'stage': 4, 'metrics': baseline,
                          'monitor_split': 'val_seen', 'monitor_metric': 'recall_at_1',
                          'monitor_value': best, 'source_checkpoint_sha256': init_sha},
                         out_dir / 'best.pt')
        print(f'BASELINE | val_seen R@1={best:.4f}; '
              f'val_unseen R@1={baseline["val_unseen"]["recall_at_1"]:.4f}', flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        # Re-mine periodically from train only. The validation set is never
        # used to choose hard pairs or anchor sampling weights.
        if epoch == start_epoch or epoch % args.remine_every == 0:
            neighbors, weights = mine_neighbors(model, index, cfg, device)
        loader = loader_for_epoch(train, cfg, neighbors, weights, epoch)
        print(f'EPOCH {epoch + 1}/{args.epochs} | batches={len(loader)} '
              f'| lrs={[g["lr"] for g in optimizer.param_groups]}', flush=True)
        train_metrics = base.train_one_epoch(model, loader, optimizer, device, cfg, scaler)
        metrics, _ = base.evaluate_splits(model, index, cfg, device,
                                          splits=('val_seen',))
        monitor = float(metrics['val_seen']['recall_at_1'])
        if not math.isfinite(monitor):
            raise FloatingPointError('Non-finite validation Recall@1')
        scheduler.step()
        improved = monitor > best + args.min_delta
        if improved:
            best, bad = monitor, 0
            full.atomic_save({'model_state_dict': model.state_dict(),
                              'config': asdict(cfg), 'epoch': epoch + 1,
                              'stage': 4, 'metrics': metrics,
                              'monitor_split': 'val_seen',
                              'monitor_metric': 'recall_at_1',
                              'monitor_value': best,
                              'source_checkpoint_sha256': init_sha},
                             out_dir / 'best.pt')
        else:
            bad += 1
        row = {'epoch': epoch + 1, 'train': train_metrics,
               'val_seen': metrics['val_seen'], 'best_val_seen_r1': best,
               'no_improvement': bad}
        history.append(row)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2))
        best_payload = torch.load(out_dir / 'best.pt', map_location='cpu', weights_only=False)
        full.atomic_save({'format': 'hard-finetune-v1',
                          'model_state_dict': model.state_dict(),
                          'config': asdict(cfg), 'dataset_kind': args.dataset_kind,
                          'index_sha256': index_hash, 'init_sha256': init_sha,
                          'epoch': epoch + 1, 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(),
                          'scaler': scaler.state_dict(), 'history': history,
                          'best_monitor': best, 'no_improvement': bad,
                          'baseline': baseline, 'best_payload': best_payload,
                          **full._rng_state(device)}, out_dir / 'last.pt')
        print(f'VALIDATION | val_seen R@1={monitor:.4f} '
              f'| best={best:.4f} | patience={bad}/{args.patience}', flush=True)
        if epoch + 1 >= args.min_epochs and bad >= args.patience:
            print('EARLY STOPPING', flush=True)
            break
        if time.monotonic() - started > args.max_hours * 3600:
            print('SESSION BUDGET | resume from last.pt in a new Kaggle run', flush=True)
            break
    best_model, _ = base.load_trained_model(out_dir / 'best.pt', cfg.weights_path, device)
    final, details = base.evaluate_splits(best_model, index, cfg, device)
    (run_dir / 'final_metrics.json').write_text(json.dumps(final, indent=2))
    base.save_retrieval_audit(details, run_dir / 'final_audit')
    result = {'dataset_kind': args.dataset_kind, 'best_checkpoint': str(out_dir / 'best.pt'),
              'last_checkpoint': str(out_dir / 'last.pt'), 'epochs': len(history),
              'baseline': baseline,
              'final': final, 'split_summary': summary, 'init_sha256': init_sha}
    (run_dir / 'run_summary.json').write_text(json.dumps(result, indent=2))
    print('RESULT |', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset-kind', choices=('bottles', 'labels'), required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--skip-file-validation', action='store_true',
                        help='Skip the per-file crop inventory for a trusted dataset')
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--min-epochs', type=int, default=7)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--remine-every', type=int, default=4)
    parser.add_argument('--head-lr', type=float, default=2e-5)
    parser.add_argument('--backbone-lr', type=float, default=1e-6)
    parser.add_argument('--min-delta', type=float, default=0.)
    parser.add_argument('--max-hours', type=float, default=9.5)
    args = parser.parse_args()
    if args.min_epochs > args.epochs or args.warmup >= args.epochs or args.remine_every < 1:
        parser.error('Invalid epoch schedule')
    run(args)


if __name__ == '__main__':
    main()
