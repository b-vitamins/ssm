from __future__ import annotations

import math
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


class Mamba1(nn.Module):
    """Selective SSM (Mamba v1) block backed by the reference ops.

    Args:
        d_model: Model embedding dimension.
        d_state: State size used by the selective scan dynamics.
        d_conv: Depthwise convolution kernel width.
        expand: Multiplicative expansion factor for the intermediate channel count.
        dt_rank: Low-rank width for the time-step projection. ``"auto"`` follows the
            upstream ``ceil(d_model / 16)`` heuristic.
        dt_min: Lower bound when sampling the initial time-step bias.
        dt_max: Upper bound when sampling the initial time-step bias.
        dt_init_floor: Floor applied to the sampled bias to avoid vanishing steps.
        dt_init: Weight initialisation policy for the low-rank time-step projection.
        bias: Whether to include bias terms on the linear projections.
        conv_bias: Whether to include bias terms on the convolution path.
        use_fast_path: Unused flag reserved for fused-kernel integration.
        layer_idx: Layer index metadata for potential parameter sharing.
        device: Optional device for parameter initialization.
        dtype: Optional dtype for parameter initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        dt_init: str = "random",
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the reference Mamba1 block.

        Args:
            d_model: Model embedding dimension.
            d_state: State size used by the selective scan dynamics.
            d_conv: Depthwise convolution kernel width.
            expand: Multiplicative expansion factor for the intermediate channel count.
            dt_rank: Low-rank width used to parameterise the time-step projection. ``"auto"``
                mirrors the upstream heuristic ``ceil(d_model / 16)``.
            dt_min: Lower bound for sampling the initial time-step bias.
            dt_max: Upper bound for sampling the initial time-step bias.
            dt_init_floor: Floor value applied to the sampled time-step bias to avoid
                vanishing updates.
            dt_init: Strategy for initialising the time-step projection weights. Matches
                the upstream options (``"random"`` or ``"constant"``).
            bias: Whether to include bias terms on the linear projections.
            conv_bias: Whether to include bias terms on the convolution path.
            use_fast_path: Unused flag reserved for fused-kernel integration.
            layer_idx: Layer index metadata for potential parameter sharing.
            device: Optional device for parameter initialization.
            dtype: Optional dtype for parameter initialization.
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = expand * d_model
        self.layer_idx = layer_idx
        self.use_fast_path = use_fast_path

        if isinstance(dt_rank, str):
            if dt_rank != "auto":
                raise ValueError("dt_rank must be an integer or 'auto'.")
            self.dt_rank = math.ceil(d_model / 16)
        else:
            if dt_rank <= 0:
                raise ValueError(
                    "dt_rank must be positive when provided as an integer."
                )
            self.dt_rank = int(dt_rank)

        if dt_min <= 0 or dt_max <= 0:
            raise ValueError("dt_min and dt_max must be positive.")
        if dt_min > dt_max:
            raise ValueError("dt_min must be less than or equal to dt_max.")
        if dt_init_floor < 0:
            raise ValueError("dt_init_floor must be non-negative.")
        if dt_init not in {"random", "constant"}:
            raise ValueError("dt_init must be either 'random' or 'constant'.")

        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_init = dt_init

        factory_kwargs: dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.in_proj = nn.Linear(
            d_model, 2 * self.inner_dim, bias=bias, **factory_kwargs
        )
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.inner_dim,
            self.dt_rank + 2 * self.d_state,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.inner_dim, bias=False, **factory_kwargs
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.inner_dim, **factory_kwargs))

        conv_weight = torch.randn(self.inner_dim, d_conv, **factory_kwargs) * 0.02
        self.conv_weight = nn.Parameter(conv_weight)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.zeros(self.inner_dim, **factory_kwargs))
        else:
            self.register_parameter("conv_bias", None)

        # Selective scan parameterisation (broadcast to ops contract).
        self.A_log = nn.Parameter(
            torch.zeros(self.inner_dim, d_state, **factory_kwargs)
        )
        self.D = nn.Parameter(torch.ones(self.inner_dim, **factory_kwargs))

        # Time-step projection initialisation mirroring the upstream recurrence.
        dt_init_std = self.dt_rank**-0.5
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:  # random
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.inner_dim, **factory_kwargs)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv_softplus)
        setattr(self.dt_bias, "_no_reinit", True)

    def forward(
        self, hidden_states: torch.Tensor, inference_params=None, **kwargs
    ) -> torch.Tensor:
        """Compute the reference Mamba1 forward pass.

        Args:
            hidden_states: torch.Tensor shaped ``(batch, seqlen, d_model)``.
            inference_params: Optional inference helper parameters (unused for the reference path).
            **kwargs: Additional keyword arguments kept for API compatibility.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``hidden_states``.

        Raises:
            ValueError: If ``hidden_states`` does not have rank 3 with ``d_model`` features.
        """

        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError("hidden_states must have shape (B, L, d_model).")

        if isinstance(inference_params, tuple):
            conv_state, ssm_state = inference_params
            return self._step_tokens(hidden_states, conv_state, ssm_state)

        if inference_params is not None:
            return self._forward_cached(hidden_states, inference_params)

        projected, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        projected = cast(torch.Tensor, projected)
        gate = cast(torch.Tensor, gate)

        compute_dtype = _compute_dtype(hidden_states)

        conv_input = projected.permute(0, 2, 1)
        conv_out = ops.dw_causal_conv(
            conv_input,
            self.conv_weight,
            bias=self.conv_bias,
            activation="silu",
        ).to(compute_dtype)

        tokens = conv_out.permute(0, 2, 1)
        proj_tokens = self.x_proj(tokens)
        dt_low_rank, B_proj, C_proj = torch.split(
            proj_tokens, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.linear(
            dt_low_rank,
            self.dt_proj.weight.to(compute_dtype),
        ).permute(0, 2, 1)
        u = conv_out
        z = gate.permute(0, 2, 1).to(compute_dtype)
        B = B_proj.permute(0, 2, 1).unsqueeze(1).to(compute_dtype)
        C = C_proj.permute(0, 2, 1).unsqueeze(1).to(compute_dtype)

        A = -torch.exp(self.A_log.to(compute_dtype))
        scan_out = ops.selective_scan(
            u=u,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            z=z,
            dt_bias=self.dt_bias,
            softplus=True,
        )
        if isinstance(scan_out, tuple):
            output = cast(torch.Tensor, scan_out[0])
        else:
            output = cast(torch.Tensor, scan_out)

        output = output.permute(0, 2, 1)
        proj_input = output.to(self.out_proj.weight.dtype)
        projected_out = self.out_proj(proj_input)
        return projected_out.to(hidden_states.dtype)

    def _forward_cached(
        self, hidden_states: torch.Tensor, inference_params
    ) -> torch.Tensor:
        if self.layer_idx is None:
            raise RuntimeError("inference requires layer_idx to be set")
        if not hasattr(inference_params, "key_value_memory_dict"):
            inference_params.key_value_memory_dict = {}
        cache_dict = inference_params.key_value_memory_dict
        batch = hidden_states.shape[0]
        batch_start = getattr(inference_params, "batch_size_offset", 0)
        batch_end = batch_start + batch
        target_batch = max(batch_end, 0)
        cache = cache_dict.get(self.layer_idx)
        if cache is None:
            cache = self.allocate_inference_cache(
                batch_size=target_batch,
                max_seqlen=getattr(
                    inference_params,
                    "max_seqlen",
                    hidden_states.shape[1],
                ),
                dtype=hidden_states.dtype,
            )
            cache_dict[self.layer_idx] = cache
        else:
            conv_cache, ssm_cache = cache
            if conv_cache.shape[0] < target_batch or ssm_cache.shape[0] < target_batch:
                new_cache = self.allocate_inference_cache(
                    batch_size=target_batch,
                    max_seqlen=getattr(
                        inference_params,
                        "max_seqlen",
                        hidden_states.shape[1],
                    ),
                    dtype=conv_cache.dtype,
                )
                new_conv_cache, new_ssm_cache = new_cache
                new_conv_cache[: conv_cache.shape[0]].copy_(conv_cache)
                new_ssm_cache[: ssm_cache.shape[0]].copy_(ssm_cache)
                cache = new_cache
                cache_dict[self.layer_idx] = cache

        conv_cache, ssm_cache = cache
        conv_slice = conv_cache[batch_start:batch_end]
        ssm_slice = ssm_cache[batch_start:batch_end]

        return self._step_tokens(hidden_states, conv_slice, ssm_slice)

    def _step_tokens(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[1] == 0:
            return hidden_states.new_zeros(
                hidden_states.shape[0], 0, hidden_states.shape[-1]
            )

        outputs: list[torch.Tensor] = []
        for t in range(hidden_states.shape[1]):
            token = hidden_states[:, t : t + 1]
            out, _, _ = self.step(token, conv_state, ssm_state)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("step expects inputs of shape (B, 1, d_model)")

        batch = hidden_states.shape[0]
        if conv_state.shape[0] != batch or conv_state.shape[1] != self.inner_dim:
            raise ValueError("conv_state shape does not match batch or inner dim")
        if ssm_state.shape[0] != batch or ssm_state.shape[1] != self.inner_dim:
            raise ValueError("ssm_state shape does not match batch or inner dim")

        projected, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        projected = projected.squeeze(1)
        gate = gate.squeeze(1)

        compute_dtype = _compute_dtype(hidden_states)

        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = projected.to(conv_state.dtype)

        conv_history = conv_state.to(compute_dtype)
        kernel = self.conv_weight.to(compute_dtype).view(1, self.inner_dim, self.d_conv)
        conv_out = (conv_history * kernel).sum(dim=-1)
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias.to(compute_dtype)
        conv_out = F.silu(conv_out)

        proj_tokens = self.x_proj(conv_out)
        dt_low_rank, B_proj, C_proj = torch.split(
            proj_tokens, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.linear(
            dt_low_rank,
            self.dt_proj.weight.to(compute_dtype),
        )
        A = -torch.exp(self.A_log.to(compute_dtype))
        z = gate.to(compute_dtype)
        B = B_proj.unsqueeze(1).to(compute_dtype)
        C = C_proj.unsqueeze(1).to(compute_dtype)

        output = ops.selective_state_step(
            ssm_state,
            conv_out,
            delta,
            A,
            B,
            C,
            self.D,
            z=z,
            dt_bias=self.dt_bias,
            softplus=True,
        )

        proj_input = output.to(self.out_proj.weight.dtype)
        projected_out = self.out_proj(proj_input)
        return projected_out.unsqueeze(1).to(hidden_states.dtype), conv_state, ssm_state

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate decoding cache tensors for streaming inference.

        Args:
            batch_size: Maximum batch size expected during decode.
            max_seqlen: Upper bound on the decoded sequence length. Unused by the reference path.
            dtype: Optional dtype override for the cache tensors.
            **kwargs: Additional keyword arguments accepted for API parity.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(conv_state, ssm_state)`` where ``conv_state``
            is shaped ``(batch_size, expand * d_model, d_conv)`` and ``ssm_state`` is shaped
            ``(batch_size, expand * d_model, d_state)``.
        """

        del max_seqlen  # Shape-independent for the reference implementation.

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
        ssm_state = torch.zeros(
            batch_size,
            self.inner_dim,
            self.d_state,
            device=device,
            dtype=cache_dtype,
        )
        return conv_state, ssm_state
