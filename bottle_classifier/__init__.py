"""Whole-bottle DINOv3 classifiers used only by the ambiguity resolver."""

from .hf_backbone import MODEL_VARIANTS, install_huggingface_backbone_loader

__all__ = ["MODEL_VARIANTS", "install_huggingface_backbone_loader"]
