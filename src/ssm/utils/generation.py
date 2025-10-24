from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch


class GenerationStreamer(Protocol):
    """Protocol describing the minimal streamer interface used by ``decode``."""

    def put(self, token: torch.Tensor) -> None:
        """Receive a newly generated token batch."""

    def end(self) -> None:
        """Signal that generation has finished."""


@dataclass
class InferenceParams:
    """State container shared between decode helpers and mixer modules."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict[int, Any] = field(default_factory=dict)
    layer_states: dict[int, Any] = field(default_factory=dict)
    lengths_per_sample: torch.Tensor | None = None
    legacy_cache: dict[str, Any] = field(default_factory=dict)

    def reset(
        self, *, max_seqlen: int | None = None, max_batch_size: int | None = None
    ) -> None:
        """Reset offsets while optionally resizing the tracked metadata."""

        if max_seqlen is not None:
            self.max_seqlen = max_seqlen
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.batch_size_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
        self.legacy_cache.clear()

    def __getitem__(self, key: str) -> Any:
        return self.legacy_cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.legacy_cache[key] = value

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.legacy_cache.get(key, default)


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


@dataclass
class DecodingCGCache:
    """CUDA graph cache mirroring the reference Mamba integration."""

    max_batch_size: int = 0
    max_seqlen: int = 0
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    callables: dict[
        tuple[int, int],
        Callable[[torch.Tensor, torch.Tensor | None, int], torch.Tensor],
    ] = field(default_factory=dict)
    mempool: Any | None = None
    inference_params: InferenceParams | None = None
    run: Callable[[torch.Tensor, torch.Tensor | None, int], torch.Tensor] | None = None


def _stream_tokens(streamer: GenerationStreamer | None, tokens: torch.Tensor) -> None:
    """Dispatch generated tokens to ``streamer`` if provided."""

    if streamer is None:
        return
    streamer.put(tokens.detach().to(device="cpu"))


def _finalize_streamer(streamer: GenerationStreamer | None) -> None:
    if streamer is not None:
        streamer.end()


def apply_repetition_penalty(
    logits: torch.Tensor, previous_tokens: torch.Tensor, repetition_penalty: float
) -> torch.Tensor:
    """Scale logits according to the repetition penalty heuristic."""

    if repetition_penalty == 1.0:
        return logits
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    gathered = torch.gather(logits, -1, previous_tokens)
    adjusted = torch.where(
        gathered < 0,
        gathered * repetition_penalty,
        gathered / repetition_penalty,
    )
    logits.scatter_(-1, previous_tokens, adjusted)
    return logits


def _prepare_inference_params(
    model,
    batch_size: int,
    max_length: int,
    input_ids: torch.Tensor,
) -> InferenceParams | None:
    """Allocate an ``InferenceParams`` instance if the model exposes caching."""

    if not hasattr(model, "allocate_inference_cache"):
        return None

    try:
        first_param = next(model.parameters())  # type: ignore[attr-defined]
        param_dtype = first_param.dtype
        param_device = first_param.device
    except StopIteration:
        param_dtype = input_ids.dtype
        param_device = input_ids.device

    cache = model.allocate_inference_cache(  # type: ignore[attr-defined]
        batch_size=batch_size,
        max_seqlen=max_length,
        dtype=param_dtype,
    )
    if not isinstance(cache, InferenceParams):
        # Wrap legacy cache structures into the inference protocol.
        key_value_memory = getattr(cache, "key_value_memory_dict", {})
        layer_states = getattr(cache, "layer_states", {})
        lengths = getattr(cache, "lengths_per_sample", None)
        cache = InferenceParams(
            max_seqlen=max_length,
            max_batch_size=batch_size,
            key_value_memory_dict=key_value_memory,
            layer_states=layer_states,
            lengths_per_sample=lengths,
            legacy_cache=cache if isinstance(cache, dict) else {},
        )
    else:
        cache.reset(max_seqlen=max_length, max_batch_size=batch_size)

    if cache.lengths_per_sample is None:
        cache.lengths_per_sample = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=param_device,
        )
    else:
        cache.lengths_per_sample = torch.zeros(
            batch_size,
            dtype=cache.lengths_per_sample.dtype,
            device=param_device,
        )

    return cache


def _set_sequence_offset(params: InferenceParams | None, offset: int) -> None:
    if params is None:
        return
    params.seqlen_offset = offset
    if params.lengths_per_sample is not None:
        params.lengths_per_sample.fill_(offset)


def update_graph_cache(
    model,
    cache: DecodingCGCache | None,
    batch_size: int,
    prompt_length: int,
    max_length: int,
    *,
    dtype: torch.dtype | None = None,
    decoding_seqlens: tuple[int, ...] = (1,),
    n_warmups: int = 2,
) -> DecodingCGCache:
    """Populate or refresh the CUDA graph cache for decoding."""

    if not torch.cuda.is_available():  # pragma: no cover - exercised on GPU builds
        raise RuntimeError("CUDA graph decoding requires CUDA availability")

    if cache is None:
        cache = DecodingCGCache()

    param_example = next(iter(model.parameters()))
    device = param_example.device
    dtype = dtype or param_example.dtype

    if (
        (cache.device, cache.dtype) != (device, dtype)
        or batch_size > cache.max_batch_size
        or max_length > cache.max_seqlen
    ):
        cache.callables.clear()
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_length
        inference_params = model.allocate_inference_cache(batch_size, max_length, dtype)
        if not isinstance(inference_params, InferenceParams):
            raise TypeError(
                "allocate_inference_cache must return InferenceParams for CUDA graphs"
            )
        lengths = torch.full(
            (batch_size,), prompt_length, dtype=torch.int32, device=device
        )
        inference_params.lengths_per_sample = lengths
        inference_params.seqlen_offset = prompt_length
        cache.inference_params = inference_params
        cache.mempool = torch.cuda.graphs.graph_pool_handle()

    assert cache.inference_params is not None

    for decoding_seqlen in decoding_seqlens:
        key = (batch_size, decoding_seqlen)
        if key in cache.callables:
            continue
        cache.callables[key] = _capture_graph(
            model,
            cache.inference_params,
            batch_size,
            max_length,
            decoding_seqlen=decoding_seqlen,
            mempool=cache.mempool,
            n_warmups=n_warmups,
        )

    def dispatch(
        input_ids: torch.Tensor, position_ids: torch.Tensor | None, seqlen: int
    ) -> torch.Tensor:
        batch, decode_len = input_ids.shape[:2]
        return cache.callables[batch, decode_len](input_ids, position_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = prompt_length
    return cache


def _capture_graph(
    model,
    inference_params: InferenceParams,
    batch_size: int,
    max_seqlen: int,
    *,
    decoding_seqlen: int = 1,
    mempool: Any | None = None,
    n_warmups: int = 2,
):  # pragma: no cover - GPU only
    device = next(iter(model.parameters())).device
    input_ids = torch.zeros(
        (batch_size, decoding_seqlen), dtype=torch.long, device=device
    )
    position_ids = torch.zeros_like(input_ids)
    original_offset = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    if inference_params.lengths_per_sample is not None:
        inference_params.lengths_per_sample.fill_(inference_params.seqlen_offset)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(n_warmups):
            model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
            )
        stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
        ).logits

    def run(
        new_input_ids: torch.Tensor, new_position_ids: torch.Tensor | None, seqlen: int
    ) -> torch.Tensor:
        if inference_params.lengths_per_sample is not None:
            inference_params.lengths_per_sample.fill_(seqlen)
        input_ids.copy_(new_input_ids)
        if new_position_ids is not None:
            position_ids.copy_(new_position_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = original_offset
    return run


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
    streamer: GenerationStreamer | None = None,
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
        repetition_penalty: Penalty factor applied to previously generated tokens.
        eos_token_id: Optional end-of-sequence token id for early stopping.
        teacher_outputs: Optional tensor providing tokens for each decode step.
        vocab_size: Optional cap for the logits dimension prior to sampling.
        cg: If ``True`` enable CUDA graph capture; requires CUDA tensors and
            a model that exposes ``allocate_inference_cache``.
        enable_timing: If ``True`` collect simple timing metrics.
        output_scores: If ``True`` collect the per-step logits snapshots.
        streamer: Optional streaming callback implementing
            :class:`GenerationStreamer`.

    Returns:
        DecodeOutput: Structure containing ``sequences`` and optional ``scores``
        and ``timings`` fields.

    Raises:
        ValueError: If shapes are invalid or ``temperature`` is not positive.
    """

    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, seqlen)")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    batch_size, prompt_length = input_ids.shape
    if max_length <= prompt_length:
        truncated = input_ids[:, :max_length].clone()
        scores: list[torch.Tensor] | None = [] if output_scores else None
        timings = {"prefill": 0.0, "decode": 0.0} if enable_timing else None
        if streamer is not None:
            streamer.end()
        return DecodeOutput(sequences=truncated, scores=scores, timings=timings)

    steps = max_length - prompt_length
    if teacher_outputs is not None:
        if teacher_outputs.shape[0] != batch_size:
            raise ValueError("teacher_outputs must match batch size")
        if teacher_outputs.shape[1] < steps:
            raise ValueError("teacher_outputs shorter than required steps")

    inference_params = _prepare_inference_params(
        model, batch_size, max_length, input_ids
    )

    cg_cache: DecodingCGCache | None = None
    if cg:
        if inference_params is None:
            raise RuntimeError("CUDA graph decoding requires inference cache support")
        if input_ids.device.type != "cuda":
            raise RuntimeError("CUDA graph decoding requires CUDA tensors")
        cg_cache = update_graph_cache(
            model,
            getattr(model, "_decoding_cache", None),
            batch_size,
            prompt_length,
            max_length,
        )
        model._decoding_cache = cg_cache  # type: ignore[attr-defined]
        inference_params = cg_cache.inference_params
        assert inference_params is not None

    scores = [] if output_scores else None
    sequences = input_ids.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    timings = {"prefill": 0.0, "decode": 0.0} if enable_timing else None
    prefill_start = time.perf_counter() if enable_timing else None

    was_training = getattr(model, "training", False)

    _stream_tokens(streamer, input_ids)

    try:
        model.eval()
        with torch.no_grad():
            _set_sequence_offset(inference_params, 0)
            outputs = model(
                sequences,
                inference_params=inference_params,
                num_last_tokens=1,
            )
            if timings is not None and prefill_start is not None:
                timings["prefill"] = time.perf_counter() - prefill_start

            logits = outputs.logits[:, -1, :]
            decode_start = time.perf_counter() if enable_timing else None
            _set_sequence_offset(inference_params, sequences.shape[1])
            processed_tokens = sequences.shape[1]

            for step in range(steps):
                step_logits = logits
                if vocab_size is not None:
                    step_logits = step_logits[..., :vocab_size]
                working_logits = apply_repetition_penalty(
                    step_logits.clone(),
                    sequences,
                    repetition_penalty,
                )
                if scores is not None:
                    scores.append(working_logits.detach().clone())

                if teacher_outputs is not None:
                    next_tokens = teacher_outputs[:, step]
                else:
                    next_tokens = sample_from_logits(
                        working_logits,
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
                        _stream_tokens(streamer, next_tokens)
                        break

                _stream_tokens(streamer, next_tokens)

                if inference_params is not None:
                    _set_sequence_offset(inference_params, processed_tokens)

                model_inputs = next_tokens.unsqueeze(-1)
                if cg_cache is not None and cg_cache.run is not None:
                    logits = cg_cache.run(model_inputs, None, processed_tokens)
                    logits = logits[:, -1, :]
                else:
                    outputs = model(
                        model_inputs,
                        inference_params=inference_params,
                        num_last_tokens=1,
                    )
                    logits = outputs.logits[:, -1, :]

                processed_tokens += 1
                _set_sequence_offset(inference_params, processed_tokens)

            if timings is not None and decode_start is not None:
                timings["decode"] = time.perf_counter() - decode_start
    finally:
        if was_training:
            model.train()
        _finalize_streamer(streamer)

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
