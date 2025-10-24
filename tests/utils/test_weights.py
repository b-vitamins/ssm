import pytest
import torch

from ssm.utils import weights


def test_save_and_load_round_trip(tmp_path):
    config = {"vocab_size": 16, "d_model": 8}
    state_dict = {"linear.weight": torch.arange(16, dtype=torch.float32).view(4, 4)}

    weights.save_pretrained_local(config, state_dict, tmp_path)

    loaded_config = weights.load_config_hf(tmp_path)
    assert loaded_config == config

    loaded_state = weights.load_state_dict_hf(tmp_path, dtype=torch.float64)
    assert loaded_state.keys() == state_dict.keys()
    assert loaded_state["linear.weight"].dtype == torch.float64
    assert torch.allclose(
        loaded_state["linear.weight"], state_dict["linear.weight"].double()
    )


def test_load_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        weights.load_config_hf(tmp_path / "missing")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        weights.load_state_dict_hf(empty_dir)

    dummy_file = tmp_path / "file.txt"
    dummy_file.write_text("{}")
    with pytest.raises(ValueError):
        weights.load_config_hf(dummy_file)
