import torch

from ssm.utils.generation import (
    InferenceParams,
    apply_repetition_penalty,
    top_p_filter,
)


def test_top_p_filter_preserves_threshold_token():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.3]]))

    filtered = top_p_filter(logits, top_p=0.5)
    probs = torch.softmax(filtered, dim=-1)

    # The second token must remain because it is required to reach the
    # requested cumulative probability mass.
    assert torch.isfinite(filtered[0, 1])
    assert torch.isinf(filtered[0, 2])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


def test_apply_repetition_penalty_scales_logits():
    logits = torch.tensor([[0.5, -0.5]])
    previous = torch.tensor([[0, 1]])

    adjusted = apply_repetition_penalty(
        logits.clone(), previous, repetition_penalty=1.2
    )

    assert torch.allclose(adjusted[0, 0], torch.tensor(0.5 / 1.2))
    assert torch.allclose(adjusted[0, 1], torch.tensor(-0.5 * 1.2))


def test_inference_params_reset_clears_lengths():
    params = InferenceParams(
        max_seqlen=8,
        max_batch_size=2,
        lengths_per_sample=torch.ones(2, dtype=torch.int32),
    )

    params.reset(max_seqlen=12, max_batch_size=3)

    assert params.max_seqlen == 12
    assert params.max_batch_size == 3
    assert params.seqlen_offset == 0
    assert params.batch_size_offset == 0
    assert params.lengths_per_sample is not None
    assert torch.count_nonzero(params.lengths_per_sample) == 0
