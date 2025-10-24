import pytest
import torch

from ssm.models import MambaConfig, MambaLMHeadModel


def test_model_generate_contract_raises():
    cfg = MambaConfig(d_model=32, n_layer=2, vocab_size=128)
    model = MambaLMHeadModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with pytest.raises(NotImplementedError):
        _ = model.generate(input_ids=input_ids, max_length=16)
