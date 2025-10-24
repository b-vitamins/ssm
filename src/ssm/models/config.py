from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MambaConfig:
    """Configuration for Mamba-based language models.

    This config is a stable surface for model construction and serialization.

    Attributes:
        d_model: Model embedding dimension.
        d_intermediate: Intermediate dimension for MLP; set 0 to disable MLP.
        n_layer: Number of mixer layers in the backbone.
        vocab_size: Vocabulary size (can be padded to multiples as needed).
        ssm_cfg: Dict passed to Mamba blocks (e.g., choose Mamba1/Mamba2 options).
        attn_layer_idx: List of layer indices that use attention instead of SSM.
        attn_cfg: Dict passed to attention blocks when used.
        rms_norm: If True, use RMSNorm; otherwise LayerNorm.
        residual_in_fp32: If True, keep residuals in fp32 for numerical stability.
        fused_add_norm: If True, enable fused add+norm backend if available.
        pad_vocab_size_multiple: Pad `vocab_size` up to multiple of this.
        tie_embeddings: If True, tie output head weights with input embeddings.
    """

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list[int] = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
