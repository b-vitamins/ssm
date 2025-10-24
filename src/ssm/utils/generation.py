from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import torch


@dataclass
class DecodeOutput:
    """Container returned by :func:`decode`.

    Attributes:
        sequences: Generated sequences including the prompt tokens.
        scores: Optional list of per-step logits snapshots.
        timings: Optional mapping of timing categories to durations in seconds.
    """

    sequences: torch.Tensor
    scores: list[torch.Tensor] | None = None
    timings: dict[str, float] | None = None


def decode(
    input_ids: torch.Tensor,
    model,
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
    output_scores: bool = False,
    streamer: Optional[object] = None,
):
    """Run an autoregressive decoding loop using the provided ``model``.

    The implementation mirrors a lightweight version of the Hugging Face
    generation utilities while remaining backend agnostic. Only the Python
    reference path is exercised in the scaffold, but the helper prepares the
    cache via ``model.allocate_inference_cache`` so that compiled backends can
    plug in later.

    Args:
        input_ids: Prompt tensor of shape ``(batch, prompt_length)``.
        model: Module exposing ``allocate_inference_cache`` and ``forward``.
        max_length: Target total sequence length (prompt + generated tokens).
        top_k: Top-k sampling parameter. ``1`` corresponds to greedy decoding.
        top_p: Nucleus sampling parameter.
        min_p: Minimum probability filter parameter.
        temperature: Softmax temperature applied before sampling.
        repetition_penalty: Currently unused placeholder for future support.
        eos_token_id: Optional end-of-sequence token id for early stopping.
        teacher_outputs: Optional tensor providing tokens for each decode step.
        vocab_size: Optional cap for the logits dimension prior to sampling.
        cg: If ``True`` attempt to use CUDA graph capture. This is a no-op for
            the CPU/reference path but the flag is accepted for API parity.
        enable_timing: If ``True`` collect simple timing metrics.
        output_scores: If ``True`` collect the per-step logits snapshots.
        streamer: Optional streaming callback with a ``put`` method accepting
            decoded token tensors and an optional ``end`` method.

    Returns:
        DecodeOutput: Structure containing ``sequences`` and optional ``scores``
        and ``timings`` fields.

    Raises:
        ValueError: If shapes are invalid or ``temperature`` is not positive.
    """

    del cg  # CUDA graphs are not exercised in the reference implementation.

    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, seqlen)")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    batch_size, prompt_length = input_ids.shape
    if max_length <= prompt_length:
        truncated = input_ids[:, :max_length].clone()
        scores: list[torch.Tensor] | None = [] if output_scores else None
        timings = {"prefill": 0.0, "decode": 0.0} if enable_timing else None
        if streamer is not None and hasattr(streamer, "end"):
            streamer.end()
        return DecodeOutput(sequences=truncated, scores=scores, timings=timings)

    steps = max_length - prompt_length
    if teacher_outputs is not None:
        if teacher_outputs.shape[0] != batch_size:
            raise ValueError("teacher_outputs must match batch size")
        if teacher_outputs.shape[1] < steps:
            raise ValueError("teacher_outputs shorter than required steps")

    # Allocate cache if available; tolerate models without an inference cache.
    cache = None
    if hasattr(model, "allocate_inference_cache"):
        try:
            first_param = next(model.parameters())  # type: ignore[attr-defined]
            param_dtype = first_param.dtype
            param_device = first_param.device
        except StopIteration:
            param_dtype = None
            param_device = input_ids.device
        cache = model.allocate_inference_cache(  # type: ignore[attr-defined]
            batch_size=batch_size,
            max_seqlen=max_length,
            dtype=param_dtype,
        )
        if isinstance(cache, dict) and "device" not in cache:
            cache.setdefault("device", param_device)

    scores = [] if output_scores else None
    sequences = input_ids.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    timings = {"prefill": 0.0, "decode": 0.0} if enable_timing else None
    prefill_start = time.perf_counter() if enable_timing else None

    was_training = getattr(model, "training", False)

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                sequences,
                inference_params=cache,
                num_last_tokens=1,
            )
            if timings is not None and prefill_start is not None:
                timings["prefill"] = time.perf_counter() - prefill_start

            logits = outputs.logits[:, -1, :]
            decode_start = time.perf_counter() if enable_timing else None

            for step in range(steps):
                step_logits = logits
                if vocab_size is not None:
                    step_logits = step_logits[..., :vocab_size]
                if scores is not None:
                    scores.append(step_logits.detach().clone())

                if teacher_outputs is not None:
                    next_tokens = teacher_outputs[:, step]
                else:
                    next_tokens = sample_from_logits(
                        step_logits,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        temperature=temperature,
                    )

                if eos_token_id is not None:
                    next_tokens = next_tokens.masked_fill(finished, eos_token_id)

                sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)
                    if finished.all():
                        if streamer is not None and hasattr(streamer, "put"):
                            streamer.put(next_tokens)
                        break

                if streamer is not None and hasattr(streamer, "put"):
                    streamer.put(next_tokens)

                outputs = model(
                    next_tokens.unsqueeze(-1),
                    inference_params=cache,
                    num_last_tokens=1,
                )
                logits = outputs.logits[:, -1, :]

            if timings is not None and decode_start is not None:
                timings["decode"] = time.perf_counter() - decode_start
    finally:
        if was_training:
            model.train()
        if streamer is not None and hasattr(streamer, "end"):
            streamer.end()

    return DecodeOutput(sequences=sequences, scores=scores, timings=timings)


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask logits that fall outside the top-k candidates.

    Args:
        logits: Logits tensor with shape `(..., vocab_size)`.
        top_k: Number of highest probability tokens to keep.

    Returns:
        torch.Tensor: Logits with non top-k entries replaced by `-inf`.
    """

    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    _, top_indices = torch.topk(logits, top_k, dim=-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(-1, top_indices, False)
    return logits.masked_fill(mask, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Logits tensor with shape `(..., vocab_size)`.
        top_p: Cumulative probability threshold.

    Returns:
        torch.Tensor: Filtered logits respecting the nucleus constraint.
    """

    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    # Shift the mask to keep the first token whose inclusion pushes the mass
    # over the threshold, matching standard nucleus sampling.
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = logits.clone()
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Filter logits using a minimum probability threshold.

    Args:
        logits: Logits tensor with shape `(..., vocab_size)`.
        min_p: Minimum probability mass required to keep a token.

    Returns:
        torch.Tensor: Logits where entries with probability `< min_p` are masked.
    """

    if min_p is None or min_p <= 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    return logits.masked_fill(probs < min_p, float("-inf"))


def sample_from_logits(
    logits: torch.Tensor,
    *,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample token indices from logits using common decoding strategies.

    Args:
        logits: Logits tensor `(B, vocab_size)`.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        min_p: Minimum probability constraint.
        temperature: Softmax temperature.

    Returns:
        torch.Tensor: Sampled token indices `(B,)`.

    Raises:
        ValueError: If `temperature` is not positive.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive for sampling")

    if top_k <= 1 and top_p <= 0.0 and min_p <= 0.0:
        scaled = logits / temperature
        return torch.argmax(scaled, dim=-1)

    filtered = logits / temperature
    filtered = top_k_filter(filtered, top_k)
    filtered = top_p_filter(filtered, top_p)
    filtered = min_p_filter(filtered, min_p)
    probs = torch.softmax(filtered, dim=-1)
    invalid = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) == 0)
    if invalid.any():
        fallback = torch.softmax((logits / temperature), dim=-1)
        probs[invalid] = fallback[invalid]
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
