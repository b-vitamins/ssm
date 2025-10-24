from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Optional

from os import PathLike, fspath

import warnings

import torch
from torch import nn

from .config import MambaConfig
from ..modules import Block, GatedMLP, MHA, Mamba1, Mamba2
from ..utils import generation as generation_utils
from ..utils.generation import InferenceParams
from ..utils.weights import load_config_hf, load_state_dict_hf, save_pretrained_local

try:  # pragma: no cover - optional fused kernels
    from mamba_ssm.ops.triton.layer_norm import (  # pyright: ignore[reportMissingImports]
        RMSNorm as TritonRMSNorm,
        layer_norm_fn,
    )
except Exception:  # pragma: no cover - optional dependency
    TritonRMSNorm = None  # type: ignore[assignment]
    layer_norm_fn = None  # type: ignore[assignment]


CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])


@dataclass
class GenerationOutput:
    """Structured response returned by :meth:`MambaLMHeadModel.generate`.

    Attributes:
        sequences: Tensor holding generated token ids ``(batch, seqlen)``.
        scores: Optional list of per-step logits for sampling diagnostics.
    """

    sequences: torch.Tensor
    scores: list[torch.Tensor] | None = None


class _FallbackRMSNorm(nn.Module):
    """Lightweight RMSNorm used when Triton kernels are unavailable."""

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        input_dtype = x.dtype
        norm_x = x.to(torch.float32)
        rms = torch.mean(norm_x * norm_x, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(rms + self.eps)
        output = x * inv_rms.to(dtype=input_dtype)
        return output * self.weight.to(dtype=input_dtype)


def _norm_cls(rms_norm: bool) -> type[nn.Module]:
    if rms_norm:
        if TritonRMSNorm is not None:
            return TritonRMSNorm  # type: ignore[return-value]
        return _FallbackRMSNorm
    return nn.LayerNorm


def _is_rms_norm(norm: nn.Module) -> bool:
    if TritonRMSNorm is not None and isinstance(norm, TritonRMSNorm):
        return True
    return isinstance(norm, _FallbackRMSNorm)


def _init_weights(
    module: nn.Module,
    *,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
) -> None:
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if not rescale_prenorm_residual:
        return

    for name, param in module.named_parameters(recurse=False):
        if name in {"out_proj.weight", "fc2.weight"}:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            with torch.no_grad():
                param /= math.sqrt(n_residuals_per_layer * n_layer)


def _create_block(
    d_model: int,
    *,
    d_intermediate: int,
    ssm_cfg: dict[str, Any] | None,
    attn_layer_idx: set[int],
    attn_cfg: dict[str, Any] | None,
    norm_epsilon: float,
    rms_norm: bool,
    residual_in_fp32: bool,
    fused_add_norm: bool,
    layer_idx: int,
    device=None,
    dtype=None,
) -> Block:
    ssm_cfg = dict(ssm_cfg or {})
    attn_cfg = dict(attn_cfg or {})
    factory_kwargs = {"device": device, "dtype": dtype}

    if layer_idx in attn_layer_idx:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    else:
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer == "Mamba2":
            mixer_cls = partial(
                Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs
            )
        elif ssm_layer == "Mamba1":
            mixer_cls = partial(
                Mamba1, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs
            )
        else:
            raise ValueError("ssm_cfg['layer'] must be either 'Mamba1' or 'Mamba2'")

    norm_type = _norm_cls(rms_norm)

    class _Norm(norm_type):  # type: ignore[misc,valid-type]
        def __init__(self, dim: int) -> None:
            super().__init__(dim, eps=norm_epsilon)
            if device is not None or dtype is not None:
                self.to(device=device, dtype=dtype)

    if d_intermediate == 0:
        mlp_cls: Callable[[int], nn.Module] | type[nn.Identity] = nn.Identity
    else:

        class _GatedMLP(GatedMLP):
            def __init__(self, dim: int) -> None:
                super().__init__(
                    dim,
                    hidden_features=d_intermediate,
                    out_features=d_model,
                    device=device,
                    dtype=dtype,
                )

        mlp_cls = _GatedMLP

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=_Norm,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    if device is not None or dtype is not None:
        block.to(device=device, dtype=dtype)
    return block


class MixerModel(nn.Module):
    """Mixer backbone mirroring the upstream Mamba reference implementation.

    Args:
        d_model: Model embedding dimension.
        n_layer: Number of stacked mixer blocks.
        d_intermediate: Hidden dimension for the gated MLP (``0`` disables it).
        vocab_size: Vocabulary size (padded before building the embedding).
        ssm_cfg: Configuration dictionary forwarded to the SSM mixer classes.
        attn_layer_idx: Indices that should instantiate attention mixers.
        attn_cfg: Configuration dictionary forwarded to attention mixers.
        norm_epsilon: Epsilon used by the backbone norms.
        rms_norm: Whether to use RMSNorm instead of LayerNorm.
        initializer_cfg: Optional overrides for the reference weight init.
        fused_add_norm: If ``True``, enable fused residual add + norm kernels.
        residual_in_fp32: Maintain residuals in fp32 for numerical stability.
        device: Optional device for parameter initialization.
        dtype: Optional dtype for parameter initialization.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg: dict[str, Any] | None = None,
        attn_layer_idx: list[int] | None = None,
        attn_cfg: dict[str, Any] | None = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg: dict[str, Any] | None = None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        if fused_add_norm and layer_norm_fn is None:
            warnings.warn(
                "fused_add_norm requested but Triton layer norm kernels are unavailable; falling back to unfused path",
                RuntimeWarning,
                stacklevel=2,
            )
            fused_add_norm = False
        self.fused_add_norm = fused_add_norm

        self.embedding = nn.Embedding(vocab_size, d_model)
        if device is not None or dtype is not None:
            self.embedding = self.embedding.to(device=device, dtype=dtype)
        attn_layers = set(attn_layer_idx or [])
        self.layers = nn.ModuleList(
            [
                _create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layers,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        norm_cls = _norm_cls(rms_norm)
        self.norm_f = norm_cls(d_model, eps=norm_epsilon)
        if device is not None or dtype is not None:
            self.norm_f = self.norm_f.to(device=device, dtype=dtype)

        residuals_per_layer = 1 if d_intermediate == 0 else 2
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                n_residuals_per_layer=residuals_per_layer,
                **(initializer_cfg or {}),
            )
        )

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> InferenceParams:
        """Allocate per-layer caches for autoregressive decoding.

        Args:
            batch_size: Maximum batch size the cache should support.
            max_seqlen: Maximum generated sequence length.
            dtype: Optional dtype override for cache tensors.
            **kwargs: Forwarded to individual mixer cache builders.

        Returns:
            InferenceParams populated with layer-specific cache entries.
        """

        dtype = dtype or self.embedding.weight.dtype
        layer_states: dict[int, Any] = {}
        key_value_memory: dict[int, Any] = {}
        for idx, layer in enumerate(self.layers):
            mixer = getattr(layer, "mixer", None)
            if isinstance(mixer, MHA):
                key_value_memory[idx] = mixer.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            elif hasattr(layer, "allocate_inference_cache"):
                allocate = getattr(layer, "allocate_inference_cache")
                if callable(allocate):
                    layer_states[idx] = allocate(
                        batch_size, max_seqlen, dtype=dtype, **kwargs
                    )
        lengths = torch.zeros(
            batch_size, dtype=torch.int32, device=self.embedding.weight.device
        )
        return InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            layer_states=layer_states,
            key_value_memory_dict=key_value_memory,
            lengths_per_sample=lengths,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        inference_params: InferenceParams | None = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Encode ``input_ids`` through the mixer stack.

        Args:
            input_ids: Token ids shaped ``(batch, seqlen)``.
            inference_params: Optional cache object for incremental decoding.
            **mixer_kwargs: Additional arguments forwarded to mixer modules.

        Returns:
            Hidden states ``(batch, seqlen, d_model)`` after the final norm.
        """

        hidden_states = self.embedding(input_ids)
        residual = None

        layer_states = {}
        if inference_params is not None:
            layer_states = inference_params.layer_states

        for idx, layer in enumerate(self.layers):
            mixer = getattr(layer, "mixer", None)
            cache_arg = None
            if inference_params is not None:
                if isinstance(mixer, MHA):
                    inference_params.key_value_memory_dict.setdefault(
                        idx,
                        mixer.allocate_inference_cache(
                            hidden_states.shape[0],
                            inference_params.max_seqlen,
                            dtype=hidden_states.dtype,
                        ),
                    )
                    cache_arg = inference_params
                else:
                    cache_arg = layer_states.get(idx)

            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=cache_arg,
                **mixer_kwargs,
            )

        if not self.fused_add_norm:
            residual = (
                hidden_states + residual if residual is not None else hidden_states
            )
            weight = getattr(self.norm_f, "weight", None)
            norm_dtype = (
                weight.dtype
                if isinstance(weight, torch.Tensor)
                else hidden_states.dtype
            )
            hidden_states = self.norm_f(residual.to(dtype=norm_dtype))
        else:
            if layer_norm_fn is None:
                raise RuntimeError("fused layer norm backend unavailable")
            weight_tensor = getattr(self.norm_f, "weight", None)
            if not isinstance(weight_tensor, torch.Tensor):
                raise TypeError("Layer norm weight parameter is missing")
            bias_attr = getattr(self.norm_f, "bias", None)
            bias_tensor: Optional[torch.Tensor]
            bias_tensor = bias_attr if isinstance(bias_attr, torch.Tensor) else None
            eps_value = float(getattr(self.norm_f, "eps", 1e-5))
            hidden_states = layer_norm_fn(
                hidden_states,
                weight_tensor,
                bias_tensor,
                eps=eps_value,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=_is_rms_norm(self.norm_f),
            )

        if inference_params is not None:
            inference_params.seqlen_offset += input_ids.shape[1]

        return hidden_states


class MambaLMHeadModel(nn.Module):
    """Causal language model composed of Mixer blocks and a linear head.

    Args:
        config: Model configuration dataclass.
        initializer_cfg: Optional overrides for weight initialisation policy.
        device: Optional device placement for newly created parameters.
        dtype: Optional dtype override for parameter initialisation.
    """

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg: dict | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.initializer_cfg = initializer_cfg or {}

        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                n_residuals_per_layer=2 if config.d_intermediate > 0 else 1,
                **(initializer_cfg or {}),
            )
        )

        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> InferenceParams:
        """Allocate caches required for incremental decoding.

        Args:
            batch_size: Maximum batch size the cache supports.
            max_seqlen: Maximum decoded sequence length.
            dtype: Optional dtype override for cache tensors.
            **kwargs: Forwarded to the backbone allocation routine.

        Returns:
            InferenceParams pre-populated with layer caches.
        """

        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ) -> CausalLMOutput:
        """Run a forward pass through the language model.

        Args:
            input_ids: Token ids ``(batch, seqlen)``.
            position_ids: Unused positional ids retained for API parity.
            inference_params: Optional cache object for decoding.
            num_last_tokens: If greater than ``0``, only return logits for the
                final ``num_last_tokens`` positions.
            **mixer_kwargs: Forwarded to the mixer blocks (e.g., varlen controls).

        Returns:
            ``CausalLMOutput`` containing the logits tensor.

        Raises:
            ValueError: If ``input_ids`` does not have rank 2.
        """

        del position_ids
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 2D tensor of shape (batch, seqlen)")

        hidden_states = self.backbone(
            input_ids,
            inference_params=inference_params,
            **mixer_kwargs,
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:, :]
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str | PathLike[str],
        *,
        device=None,
        dtype=None,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        strict: bool = True,
    ) -> "MambaLMHeadModel":
        """Instantiate a model from a local directory or Hub checkpoint.

        Args:
            pretrained_model_name: Local path or Hugging Face repo id.
            device: Optional device placement for the loaded weights.
            dtype: Optional dtype conversion applied to the loaded weights.
            revision: Optional Git revision used when fetching from the Hub.
            token: Optional Hugging Face token for private repositories.
            cache_dir: Optional cache directory override for Hub downloads.
            local_files_only: If ``True``, avoid network access when resolving.
            strict: Passed to :meth:`torch.nn.Module.load_state_dict`.

        Returns:
            Loaded :class:`MambaLMHeadModel` instance.
        """

        model_name = fspath(pretrained_model_name)
        cache_dir_str = fspath(cache_dir) if cache_dir is not None else None
        config_data = load_config_hf(
            model_name,
            revision=revision,
            token=token,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
        )
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype)
        state_dict = load_state_dict_hf(
            model_name,
            device=device,
            dtype=dtype,
            revision=revision,
            token=token,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
        )
        model.load_state_dict(state_dict, strict=strict)
        return model

    def save_pretrained(self, save_directory: str | PathLike[str]) -> None:
        """Persist the model weights and configuration to ``save_directory``.

        Args:
            save_directory: Target directory that will receive the checkpoint.
        """

        save_pretrained_local(
            asdict(self.config), self.state_dict(), fspath(save_directory)
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        top_k: int = 1,
        top_p: float = 0.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        teacher_outputs: torch.Tensor | None = None,
        vocab_size: int | None = None,
        cg: bool = False,
        enable_timing: bool = False,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        streamer: generation_utils.GenerationStreamer | None = None,
        **kwargs,
    ):
        """Generate tokens via greedy or sampling-based decoding.

        Args:
            input_ids: Prompt token ids ``(batch, prompt_length)``.
            max_length: Target total length (prompt plus generated tokens).
            top_k: Top-k sampling parameter; ``1`` performs greedy decoding.
            top_p: Top-p (nucleus) sampling parameter.
            min_p: Min-p sampling parameter.
            temperature: Softmax temperature applied to logits.
            repetition_penalty: Penalize repeated tokens when sampling.
            eos_token_id: Optional id that triggers early stop when generated.
            teacher_outputs: Optional teacher forcing tokens per decode step.
            vocab_size: Optional logits truncation size before sampling.
            cg: Enable CUDA graph caching for compatible GPU execution.
            enable_timing: If ``True``, capture simple timing breakdowns.
            return_dict_in_generate: If ``True``, return a :class:`GenerationOutput`.
            output_scores: If ``True``, include per-step logits in the output.
            streamer: Optional callback invoked with prompt and sampled tokens.
            **kwargs: Reserved for future extensions.

        Returns:
            Either a tensor of generated sequences or a tuple/``GenerationOutput``
            depending on ``return_dict_in_generate`` and ``output_scores``.

        Raises:
            ValueError: If ``input_ids`` rank is not 2 or ``max_length`` is
                non-positive.
        """

        decode_output = generation_utils.decode(
            input_ids=input_ids,
            model=self,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            teacher_outputs=teacher_outputs,
            vocab_size=vocab_size,
            cg=cg,
            enable_timing=enable_timing,
            output_scores=output_scores,
            streamer=streamer,
            **kwargs,
        )

        output = GenerationOutput(
            sequences=decode_output.sequences, scores=decode_output.scores
        )
        if return_dict_in_generate:
            return output
        if output_scores:
            return output.sequences, output.scores
        return output.sequences
