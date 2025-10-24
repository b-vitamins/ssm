import torch

from ssm.models import MambaConfig, MambaLMHeadModel


def test_model_generate_contract_returns_sequences():
    cfg = MambaConfig(d_model=32, n_layer=2, vocab_size=64)
    model = MambaLMHeadModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 4))
    max_length = 6

    sequences = model.generate(input_ids=input_ids, max_length=max_length)

    assert isinstance(sequences, torch.Tensor)
    assert sequences.shape == (2, max_length)
    assert torch.equal(sequences[:, : input_ids.size(1)], input_ids)
