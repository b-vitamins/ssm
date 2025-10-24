from __future__ import annotations

from typing import Any

import torch

from ssm.models import MambaConfig, MambaLMHeadModel
from ssm.models.lm import GenerationOutput
from ssm.utils.generation import InferenceParams
from ssm.utils import generation as generation_utils


def _make_config(**overrides: Any) -> MambaConfig:
    base: dict[str, Any] = {
        "d_model": 8,
        "d_intermediate": 16,
        "n_layer": 2,
        "vocab_size": 16,
        "tie_embeddings": False,
        "ssm_cfg": {"layer": "Mamba1", "d_state": 4, "d_conv": 2, "expand": 2},
        "attn_layer_idx": [],
        "attn_cfg": {},
        "rms_norm": False,
        "residual_in_fp32": False,
        "fused_add_norm": False,
        "pad_vocab_size_multiple": 1,
    }
    base.update(overrides)
    return MambaConfig(**base)


def _zero_model_parameters(model: MambaLMHeadModel) -> None:
    for parameter in model.parameters():
        parameter.data.zero_()


def test_generate_greedy_cpu():
    torch.manual_seed(0)
    cfg = _make_config()
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
    cfg = _make_config(vocab_size=10)
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
    cfg = _make_config(vocab_size=12)
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


def test_allocate_inference_cache_with_attention():
    cfg = _make_config(
        attn_layer_idx=[0],
        attn_cfg={"num_heads": 2, "num_heads_kv": 1, "causal": True},
    )
    model = MambaLMHeadModel(cfg)
    cache = model.allocate_inference_cache(
        batch_size=2, max_seqlen=6, dtype=torch.float32
    )

    assert isinstance(cache, InferenceParams)
    assert cache.max_seqlen == 6
    assert 1 in cache.layer_states
    conv_state, ssm_state = cache.layer_states[1]
    ssm_cfg = cfg.ssm_cfg
    expand = int(ssm_cfg.get("expand", 1))
    d_conv = int(ssm_cfg.get("d_conv", 0))
    d_state = int(ssm_cfg.get("d_state", 0))
    assert conv_state.shape == (2, cfg.d_model * expand, d_conv)
    assert ssm_state.shape == (2, cfg.d_model * expand, d_state)

    assert 0 in cache.key_value_memory_dict
    kv_cache, conv_cache = cache.key_value_memory_dict[0]
    assert kv_cache.shape[:2] == (2, 6)
    assert conv_cache is None


def test_save_and_load_pretrained_round_trip(tmp_path):
    torch.manual_seed(42)
    cfg = _make_config()
    model = MambaLMHeadModel(cfg)
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.01)

    model.save_pretrained(tmp_path)
    reloaded = MambaLMHeadModel.from_pretrained(tmp_path)

    assert reloaded.config == model.config
    original_state = model.state_dict()
    loaded_state = reloaded.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for name, tensor in original_state.items():
        assert torch.allclose(tensor, loaded_state[name])
