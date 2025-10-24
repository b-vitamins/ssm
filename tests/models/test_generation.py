from __future__ import annotations

import torch

from ssm.models import MambaConfig, MambaLMHeadModel
from ssm.utils import generation as generation_utils
from ssm.models.lm import GenerationOutput


def _zero_model_parameters(model: MambaLMHeadModel) -> None:
    for parameter in model.parameters():
        parameter.data.zero_()


def test_generate_greedy_cpu():
    torch.manual_seed(0)
    cfg = MambaConfig(d_model=8, n_layer=1, vocab_size=16, tie_embeddings=False)
    model = MambaLMHeadModel(cfg)
    _zero_model_parameters(model)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    max_length = 5

    result = model.generate(input_ids=input_ids, max_length=max_length, top_k=1)
    assert isinstance(result, torch.Tensor)
    sequences = result

    assert sequences.shape == (2, max_length)
    assert torch.all(sequences[:, : input_ids.size(1)] == input_ids)
    assert torch.all(sequences[:, input_ids.size(1) :] == 0)


def test_generate_sampling_top_k():
    torch.manual_seed(1)
    cfg = MambaConfig(d_model=8, n_layer=1, vocab_size=10, tie_embeddings=False)
    model = MambaLMHeadModel(cfg)
    _zero_model_parameters(model)

    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    max_length = 4

    result = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        top_k=4,
        temperature=1.0,
    )
    assert isinstance(result, torch.Tensor)
    sequences = result

    generated = sequences[:, input_ids.size(1) :]
    assert generated.shape == (1, max_length - input_ids.size(1))
    baseline_logits = torch.zeros((1, cfg.vocab_size))
    filtered = generation_utils.top_k_filter(baseline_logits, 4)
    valid = torch.isfinite(filtered[0])
    assert torch.all(valid[generated[0]])


def test_generate_returns_scores():
    torch.manual_seed(0)
    cfg = MambaConfig(d_model=8, n_layer=1, vocab_size=12, tie_embeddings=False)
    model = MambaLMHeadModel(cfg)
    _zero_model_parameters(model)

    input_ids = torch.tensor([[2, 3]], dtype=torch.long)
    output = model.generate(
        input_ids=input_ids,
        max_length=3,
        output_scores=True,
        return_dict_in_generate=True,
    )
    assert isinstance(output, GenerationOutput)
    result = output

    assert result.sequences.shape == (1, 3)
    assert result.scores is not None
    assert len(result.scores) == 1
    assert result.scores[0].shape == (1, cfg.vocab_size)

    baseline_logits = torch.zeros((1, cfg.vocab_size))
    expected_token = generation_utils.sample_from_logits(
        baseline_logits,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
    )
    assert torch.equal(result.sequences[:, -1], expected_token)
