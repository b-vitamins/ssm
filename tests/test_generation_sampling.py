import torch

from ssm.utils.generation import top_p_filter


def test_top_p_filter_preserves_threshold_token():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.3]]))

    filtered = top_p_filter(logits, top_p=0.5)
    probs = torch.softmax(filtered, dim=-1)

    # The second token must remain because it is required to reach the
    # requested cumulative probability mass.
    assert torch.isfinite(filtered[0, 1])
    assert torch.isinf(filtered[0, 2])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
