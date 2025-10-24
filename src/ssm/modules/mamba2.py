from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from ssm import ops


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return the accumulation dtype for a given tensor."""

    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


class Mamba2(nn.Module):
    """State Space Dual (Mamba v2) block implemented with reference ops.

    Args:
        d_model: Model embedding dimension.
        d_state: State size used by the selective scan dynamics.
        d_conv: Depthwise convolution kernel width.
        headdim: Channel dimension per attention-style head.
        expand: Multiplicative expansion factor for the intermediate channel count.
        ngroups: Number of groups for grouped convolutions (reserved for fused kernels).
        d_ssm: Number of channels routed through the SSM branch.
        chunk_size: Scan chunk size used by the reference ops implementation.
        bias: Whether to include bias terms on the linear projections.
        conv_bias: Whether to include bias terms on the convolution path.
        layer_idx: Layer index metadata for potential parameter sharing.
        device: Optional device for parameter initialization.
        dtype: Optional dtype for parameter initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        d_ssm: int | None = None,
        chunk_size: int = 256,
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the reference Mamba2 block.

        Args:
            d_model: Model embedding dimension.
            d_state: State size used by the selective scan dynamics.
            d_conv: Depthwise convolution kernel width.
            headdim: Channel dimension per attention-style head.
            expand: Multiplicative expansion factor for the intermediate channel count.
            ngroups: Number of groups for grouped convolutions (reserved for fused kernels).
            d_ssm: Number of channels routed through the SSM branch.
            chunk_size: Scan chunk size used by the reference ops implementation.
            bias: Whether to include bias terms on the linear projections.
            conv_bias: Whether to include bias terms on the convolution path.
            layer_idx: Layer index metadata for potential parameter sharing.
            device: Optional device for parameter initialization.
            dtype: Optional dtype for parameter initialization.
        """
        super().__init__()

        if expand * d_model % headdim != 0:
            raise ValueError("expand * d_model must be divisible by headdim.")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.expand = expand
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.inner_dim = expand * d_model
        self.nheads = self.inner_dim // headdim
        self.ssm_dim = d_ssm if d_ssm is not None else self.inner_dim
        if self.ssm_dim < 0 or self.ssm_dim > self.inner_dim:
            raise ValueError("d_ssm must lie within [0, expand * d_model].")
        if self.ssm_dim % headdim != 0:
            raise ValueError("d_ssm must be divisible by headdim for head packing.")
        self.ssm_heads = self.ssm_dim // headdim
        self.mlp_dim = self.inner_dim - self.ssm_dim

        factory_kwargs: dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.in_proj = nn.Linear(d_model, self.inner_dim, bias=bias, **factory_kwargs)
        self.gate_proj = nn.Linear(d_model, self.inner_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias, **factory_kwargs)

        conv_weight = torch.randn(self.inner_dim, d_conv, **factory_kwargs) * 0.02
        self.conv_weight = nn.Parameter(conv_weight)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.zeros(self.inner_dim, **factory_kwargs))
        else:
            self.register_parameter("conv_bias", None)

        # Per-head SSM parameters broadcast by the chunked scan.
        self.dt_proj_weight = nn.Parameter(
            torch.randn(self.nheads, headdim, **factory_kwargs) * 0.02
        )
        self.dt_proj_bias = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
        self.A_log = nn.Parameter(torch.zeros(self.nheads, headdim, **factory_kwargs))
        self.B = nn.Parameter(
            torch.randn(self.nheads, headdim, **factory_kwargs) * 0.02
        )
        self.C = nn.Parameter(
            torch.randn(self.nheads, headdim, **factory_kwargs) * 0.02
        )
        self.D = nn.Parameter(torch.ones(self.nheads, headdim, **factory_kwargs))

        if self.mlp_dim > 0:
            self.mlp = nn.Linear(
                self.mlp_dim, self.mlp_dim, bias=bias, **factory_kwargs
            )
        else:
            self.register_module("mlp", None)

    def _forward_impl(
        self, hidden_states: torch.Tensor, seq_lens: torch.Tensor | None
    ) -> torch.Tensor:
        """Evaluate the dense Mamba2 path.

        Args:
            hidden_states: torch.Tensor shaped ``(batch, seqlen, d_model)``.
            seq_lens: Optional torch.Tensor of per-sequence lengths used to mask outputs.

        Returns:
            torch.Tensor: Output tensor shaped ``(batch, seqlen, d_model)``.

        Raises:
            ValueError: If ``hidden_states`` does not have rank 3 with ``d_model`` features.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError("hidden_states must have shape (B, L, d_model).")

        batch, seqlen, _ = hidden_states.shape

        compute_dtype = _compute_dtype(hidden_states)

        projected = self.in_proj(hidden_states)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).to(compute_dtype)

        conv_input = projected.permute(0, 2, 1)
        conv_out_cf = ops.dw_causal_conv(
            conv_input,
            self.conv_weight,
            bias=self.conv_bias,
            activation="silu",
        ).to(compute_dtype)

        lengths_tensor: torch.Tensor | None = None
        if self.ssm_dim > 0:
            ssm_cf = conv_out_cf[:, : self.ssm_dim].contiguous()
            ssm_heads = ssm_cf.view(batch, self.ssm_heads, self.headdim, seqlen)
            X = ssm_heads.permute(0, 3, 1, 2).contiguous()
            dt = torch.einsum(
                "bhpl,hp->bhl",
                ssm_heads.to(compute_dtype),
                self.dt_proj_weight[: self.ssm_heads].to(compute_dtype),
            )
            dt = dt + self.dt_proj_bias[: self.ssm_heads].to(compute_dtype).view(
                1, self.ssm_heads, 1
            )
            dt = F.softplus(dt.transpose(1, 2))

            seq_meta = None
            if seq_lens is not None:
                lengths_tensor = seq_lens.to(
                    dtype=torch.long, device=hidden_states.device
                )
                seq_meta = {"seq_lens": lengths_tensor.tolist()}

            ssm_out = ops.ssd_chunk_scan(
                X=X,
                dt=dt,
                A=-torch.exp(self.A_log[: self.ssm_heads].to(compute_dtype)),
                B=self.B[: self.ssm_heads],
                C=self.C[: self.ssm_heads],
                chunk_size=self.chunk_size,
                D=self.D[: self.ssm_heads],
                z=None,
                seq_meta=seq_meta,
            )
            ssm_out = ssm_out.view(batch, seqlen, self.ssm_dim).to(compute_dtype)
        else:
            ssm_out = hidden_states.new_zeros(batch, seqlen, 0).to(compute_dtype)
            if seq_lens is not None:
                lengths_tensor = seq_lens.to(
                    dtype=torch.long, device=hidden_states.device
                )

        if self.mlp_dim > 0:
            mlp_cf = conv_out_cf[:, self.ssm_dim :]
            mlp_in = mlp_cf.permute(0, 2, 1).to(self.mlp.weight.dtype)
            mlp_proj = self.mlp(mlp_in)
            mlp_out = F.silu(mlp_proj).to(compute_dtype)
        else:
            mlp_out = hidden_states.new_zeros(batch, seqlen, 0).to(compute_dtype)

        combined = torch.cat([ssm_out.to(compute_dtype), mlp_out], dim=-1)
        combined = combined * gate

        if seq_lens is not None:
            assert lengths_tensor is not None
            mask = torch.arange(seqlen, device=hidden_states.device).unsqueeze(
                0
            ).expand(batch, seqlen) < lengths_tensor.unsqueeze(1)
            combined = combined * mask.unsqueeze(-1).to(combined.dtype)

        proj_input = combined.to(self.out_proj.weight.dtype)
        projected = self.out_proj(proj_input)
        return projected.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Route tokens through the reference Mamba2 implementation.

        Args:
            hidden_states: torch.Tensor in dense ``(batch, seqlen, d_model)`` or routed
                ``(tokens, d_model)``/``(1, tokens, d_model)`` format.
            inference_params: Optional inference helper parameters (not yet supported).
            seq_idx: Optional torch.Tensor of routing indices describing which sequence each token
                belongs to.
            cu_seqlens: Optional torch.Tensor of exclusive prefix sums describing per-sequence lengths.
            **kwargs: Additional keyword arguments kept for API compatibility.

        Returns:
            torch.Tensor: Output tensor matching the layout of ``hidden_states``.

        Raises:
            ValueError: If the provided routing metadata is invalid for the given inputs.
        """
        if isinstance(inference_params, dict):
            return self._forward_with_cache_dict(
                hidden_states,
                inference_params,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
            )

        if inference_params is not None:
            return self._forward_inference(
                hidden_states,
                inference_params,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
            )

        if cu_seqlens is None:
            if seq_idx is not None:
                raise ValueError("seq_idx provided without cu_seqlens metadata.")
            return self._forward_impl(hidden_states, seq_lens=None)

        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1-D tensor of length B + 1.")

        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch = seq_lens.numel()
        total_tokens = int(seq_lens.sum().item())

        def _normalize_seq_idx(idx: torch.Tensor) -> torch.Tensor:
            if idx.ndim == 2:
                if idx.shape[0] != 1:
                    raise ValueError(
                        "seq_idx must broadcast with the flattened batch dimension."
                    )
                idx = idx[0]
            if idx.ndim != 1:
                raise ValueError(
                    "seq_idx must be rank-1 (or (1, T) for cu_seqlens routing)."
                )
            if idx.numel() != total_tokens:
                raise ValueError(
                    "seq_idx must have the same number of tokens as hidden_states."
                )
            idx = idx.to(device=hidden_states.device, dtype=torch.long)
            if torch.any(idx < 0) or torch.any(idx >= batch):
                raise ValueError("seq_idx entries must index sequences in [0, B).")
            return idx

        if hidden_states.ndim == 3 and hidden_states.shape[0] == batch:
            return self._forward_impl(hidden_states, seq_lens=seq_lens)

        if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
            if hidden_states.shape[1] != total_tokens:
                raise ValueError("Flattened tokens must match cu_seqlens total length.")
            flat = hidden_states[0]
            keep_batch_dim = True
        elif hidden_states.ndim == 2:
            if hidden_states.shape[0] != total_tokens:
                raise ValueError("Flattened tokens must match cu_seqlens total length.")
            flat = hidden_states
            keep_batch_dim = False
        else:
            raise ValueError("Unsupported hidden_states layout for varlen routing.")

        if seq_idx is not None:
            seq_idx = _normalize_seq_idx(seq_idx)

        max_len = int(seq_lens.max().item()) if batch > 0 else 0
        padded = hidden_states.new_zeros(batch, max_len, self.d_model)

        scatter_positions: list[torch.Tensor] = []
        cursor = 0
        for b, length in enumerate(seq_lens.tolist()):
            if length == 0:
                scatter_positions.append(
                    torch.empty(0, dtype=torch.long, device=hidden_states.device)
                )
                continue
            if seq_idx is None:
                positions = torch.arange(
                    cursor,
                    cursor + length,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            else:
                positions = (seq_idx == b).nonzero(as_tuple=False).squeeze(-1)
                if positions.numel() != length:
                    raise ValueError(
                        "seq_idx counts must agree with cu_seqlens lengths."
                    )
            padded[b, :length] = flat.index_select(0, positions)
            scatter_positions.append(positions)
            if seq_idx is None:
                cursor += length

        dense_out = self._forward_impl(padded, seq_lens=seq_lens)

        routed = flat.new_zeros(total_tokens, self.d_model)
        for b, length in enumerate(seq_lens.tolist()):
            if length == 0:
                continue
            positions = scatter_positions[b]
            routed.index_copy_(0, positions, dense_out[b, :length])

        if keep_batch_dim:
            return routed.unsqueeze(0)
        return routed

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Allocate decoding cache tensors for streaming inference.

        Args:
            batch_size: Maximum batch size expected during decode.
            max_seqlen: Upper bound on the decoded sequence length. Unused by the reference path.
            dtype: Optional dtype override for the cache tensors.
            **kwargs: Additional keyword arguments accepted for API parity.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing ``conv_state`` with shape
            ``(batch_size, expand * d_model, d_conv)`` and ``ssd_state`` with shape
            ``(batch_size, ssm_heads, headdim)``.
        """

        del max_seqlen

        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        cache_dtype = dtype or (
            param.dtype if param is not None else torch.get_default_dtype()
        )

        conv_state = torch.zeros(
            batch_size,
            self.inner_dim,
            self.d_conv,
            device=device,
            dtype=cache_dtype,
        )
        ssd_state = torch.zeros(
            batch_size,
            self.ssm_heads,
            self.headdim,
            device=device,
            dtype=cache_dtype,
        )
        return {"conv_state": conv_state, "ssd_state": ssd_state}

    def _forward_with_cache_dict(
        self,
        hidden_states: torch.Tensor,
        cache: dict[str, torch.Tensor],
        *,
        seq_idx: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        conv_state = cache["conv_state"]
        ssd_state = cache["ssd_state"]
        if cu_seqlens is None:
            return self._step_dense_tokens(hidden_states, conv_state, ssd_state)
        return self._step_varlen_tokens(
            hidden_states,
            cu_seqlens,
            seq_idx,
            conv_state,
            ssd_state,
        )

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        inference_params,
        *,
        seq_idx: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.layer_idx is None:
            raise RuntimeError("inference requires layer_idx to be set")
        if not hasattr(inference_params, "key_value_memory_dict"):
            inference_params.key_value_memory_dict = {}
        cache_dict = cast(
            dict[int, dict[str, torch.Tensor]],
            inference_params.key_value_memory_dict,
        )

        if cu_seqlens is None:
            if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
                raise ValueError("hidden_states must have shape (B, L, d_model).")
            batch = hidden_states.shape[0]
        else:
            if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                raise ValueError("cu_seqlens must be a 1-D tensor of length B + 1.")
            batch = cu_seqlens.numel() - 1

        batch_start = getattr(inference_params, "batch_size_offset", 0)
        batch_end = batch_start + batch
        target_batch = max(batch_end, 0)

        cache = cache_dict.get(self.layer_idx)
        if cache is None or (
            cache["conv_state"].shape[0] < target_batch
            or cache["ssd_state"].shape[0] < target_batch
        ):
            cache = self.allocate_inference_cache(
                batch_size=target_batch,
                max_seqlen=getattr(
                    inference_params,
                    "max_seqlen",
                    hidden_states.shape[1] if hidden_states.ndim == 3 else 1,
                ),
                dtype=hidden_states.dtype,
            )
            cache_dict[self.layer_idx] = cache

        conv_cache = cache["conv_state"]
        ssd_cache = cache["ssd_state"]
        conv_slice = conv_cache[batch_start:batch_end]
        ssd_slice = ssd_cache[batch_start:batch_end]

        if cu_seqlens is None:
            return self._step_dense_tokens(hidden_states, conv_slice, ssd_slice)

        return self._step_varlen_tokens(
            hidden_states,
            cu_seqlens,
            seq_idx,
            conv_slice,
            ssd_slice,
        )

    def _step_dense_tokens(
        self,
        tokens: torch.Tensor,
        conv_state: torch.Tensor,
        ssd_state: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens.new_zeros(tokens.shape[0], 0, self.d_model)
        outputs: list[torch.Tensor] = []
        for t in range(tokens.shape[1]):
            out, _, _ = self.step(tokens[:, t : t + 1], conv_state, ssd_state)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def _step_varlen_tokens(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_idx: torch.Tensor | None,
        conv_state: torch.Tensor,
        ssd_state: torch.Tensor,
    ) -> torch.Tensor:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch = seq_lens.numel()
        total_tokens = int(seq_lens.sum().item())

        if hidden_states.ndim == 3 and hidden_states.shape[0] == batch:
            if seq_idx is not None:
                raise ValueError(
                    "seq_idx cannot be provided with dense varlen hidden_states."
                )
            max_len = hidden_states.shape[1]
            outputs = hidden_states.new_zeros(batch, max_len, self.d_model)
            for b, length in enumerate(seq_lens.tolist()):
                if length == 0:
                    continue
                seq_tokens = hidden_states[b : b + 1, :length]
                seq_out = self._step_dense_tokens(
                    seq_tokens, conv_state[b : b + 1], ssd_state[b : b + 1]
                )
                outputs[b, :length] = seq_out[0]
            return outputs

        if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
            flat = hidden_states[0]
            keep_batch_dim = True
        elif hidden_states.ndim == 2:
            flat = hidden_states
            keep_batch_dim = False
        else:
            raise ValueError(
                "Unsupported hidden_states layout for varlen routing with cache."
            )

        if flat.shape[0] != total_tokens:
            raise ValueError("Flattened tokens must match cu_seqlens total length.")

        device = hidden_states.device

        def _normalize_seq_idx(idx: torch.Tensor) -> torch.Tensor:
            if idx.ndim == 2:
                if idx.shape[0] != 1:
                    raise ValueError(
                        "seq_idx must broadcast with the flattened batch dimension."
                    )
                idx = idx[0]
            if idx.ndim != 1:
                raise ValueError(
                    "seq_idx must be rank-1 (or (1, T) for cu_seqlens routing)."
                )
            if idx.numel() != total_tokens:
                raise ValueError(
                    "seq_idx must have the same number of tokens as hidden_states."
                )
            idx = idx.to(device=device, dtype=torch.long)
            if torch.any(idx < 0) or torch.any(idx >= batch):
                raise ValueError("seq_idx entries must index sequences in [0, B).")
            return idx

        positions: list[torch.Tensor] = []
        if seq_idx is None:
            cursor = 0
            for length in seq_lens.tolist():
                pos = torch.arange(
                    cursor,
                    cursor + length,
                    device=device,
                    dtype=torch.long,
                )
                positions.append(pos)
                cursor += length
        else:
            seq_idx = _normalize_seq_idx(seq_idx)
            for b, length in enumerate(seq_lens.tolist()):
                pos = (seq_idx == b).nonzero(as_tuple=False).squeeze(-1)
                if pos.numel() != length:
                    raise ValueError(
                        "seq_idx counts must agree with cu_seqlens lengths."
                    )
                positions.append(pos)

        flat_out = flat.new_zeros(total_tokens, self.d_model)
        for b, length in enumerate(seq_lens.tolist()):
            if length == 0:
                continue
            pos = positions[b]
            seq_tokens = flat.index_select(0, pos).unsqueeze(0)
            seq_out = self._step_dense_tokens(
                seq_tokens, conv_state[b : b + 1], ssd_state[b : b + 1]
            )
            flat_out.index_copy_(0, pos, seq_out.squeeze(0))

        if keep_batch_dim:
            return flat_out.unsqueeze(0)
        return flat_out

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssd_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("step expects inputs of shape (B, 1, d_model)")

        batch = hidden_states.shape[0]
        if conv_state.shape[0] != batch or conv_state.shape[1] != self.inner_dim:
            raise ValueError("conv_state shape does not match batch or inner dim")
        if ssd_state.shape[0] != batch or ssd_state.shape[1] != self.ssm_heads:
            raise ValueError("ssd_state shape does not match batch or head count")

        compute_dtype = _compute_dtype(hidden_states)

        projected = self.in_proj(hidden_states).squeeze(1)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).squeeze(1)

        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = projected.to(conv_state.dtype)

        conv_hist = conv_state.to(compute_dtype)
        kernel = self.conv_weight.to(compute_dtype).view(1, self.inner_dim, self.d_conv)
        conv_out = (conv_hist * kernel).sum(dim=-1)
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias.to(compute_dtype)
        conv_out = F.silu(conv_out)

        if self.ssm_dim > 0:
            ssm_drive = conv_out[:, : self.ssm_dim].view(
                batch, self.ssm_heads, self.headdim
            )
            dt_raw = torch.einsum(
                "bhp,hp->bh",
                ssm_drive.to(compute_dtype),
                self.dt_proj_weight[: self.ssm_heads].to(compute_dtype),
            )
            dt = F.softplus(
                dt_raw + self.dt_proj_bias[: self.ssm_heads].to(compute_dtype)
            )

            decay = torch.exp(
                dt.unsqueeze(-1)
                * -torch.exp(self.A_log[: self.ssm_heads].to(compute_dtype))
            )

            drive = self.B[: self.ssm_heads].to(compute_dtype) * ssm_drive.to(
                compute_dtype
            )

            state_compute = ssd_state.to(compute_dtype)
            new_state = decay * state_compute + dt.unsqueeze(-1) * drive
            ssd_state.copy_(new_state.to(ssd_state.dtype))

            proj = self.C[: self.ssm_heads].to(compute_dtype)
            ssm_out_heads = new_state * proj
            if self.D is not None:
                ssm_out_heads = ssm_out_heads + (
                    self.D[: self.ssm_heads].to(compute_dtype)
                    * ssm_drive.to(compute_dtype)
                )
            ssm_out = ssm_out_heads.reshape(batch, self.ssm_dim)
        else:
            ssm_out = conv_out.new_zeros(batch, 0)

        if self.mlp_dim > 0:
            mlp_drive = conv_out[:, self.ssm_dim :]
            mlp_proj = self.mlp(mlp_drive.to(self.mlp.weight.dtype))
            mlp_out = F.silu(mlp_proj).to(compute_dtype)
        else:
            mlp_out = conv_out.new_zeros(batch, 0)

        combined = torch.cat([ssm_out.to(compute_dtype), mlp_out], dim=-1)
        combined = combined * gate.to(compute_dtype)

        proj_input = combined.to(self.out_proj.weight.dtype)
        projected_out = self.out_proj(proj_input)
        return projected_out.unsqueeze(1).to(hidden_states.dtype), conv_state, ssd_state
