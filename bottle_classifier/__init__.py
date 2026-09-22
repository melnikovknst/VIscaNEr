"""Whole-bottle DINOv3 classifiers used only by the ambiguity resolver."""

from .local_backbone import MODEL_VARIANTS, install_local_backbone_loader

__all__ = ["MODEL_VARIANTS", "install_local_backbone_loader"]
