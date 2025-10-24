import json
import sys
import types
from typing import Any

import pytest
import torch

from ssm.utils import weights


def _register_fake_hub(monkeypatch, cached_file):
    module = types.ModuleType("huggingface_hub")
    module.cached_file = cached_file  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


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


def test_load_config_hf_remote(monkeypatch, tmp_path):
    config_data = {"foo": "bar"}
    remote_path = tmp_path / weights.CONFIG_FILE
    remote_path.write_text(json.dumps(config_data))

    def fake_cached_file(repo_id, filename, **kwargs):
        assert repo_id == "state-spaces/test"
        assert filename == weights.CONFIG_FILE
        return str(remote_path)

    _register_fake_hub(monkeypatch, fake_cached_file)

    loaded = weights.load_config_hf("state-spaces/test")
    assert loaded == config_data


def test_load_state_dict_hf_remote_dtype_move(monkeypatch, tmp_path):
    weights_path = tmp_path / weights.WEIGHTS_FILE
    torch.save({"param": torch.ones(2, dtype=torch.float32)}, weights_path)

    calls: dict[str, Any] = {}

    def fake_cached_file(repo_id, filename, **kwargs):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["kwargs"] = kwargs
        return str(weights_path)

    _register_fake_hub(monkeypatch, fake_cached_file)

    original_load = torch.load

    def capture_load(path, *args, **kwargs):
        calls["map_location"] = kwargs.get("map_location") if kwargs else None
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(weights.torch, "load", capture_load)

    state = weights.load_state_dict_hf(
        "state-spaces/test", device="cpu", dtype=torch.float16
    )

    assert calls["repo_id"] == "state-spaces/test"
    assert calls["filename"] == weights.WEIGHTS_FILE
    assert calls["kwargs"]["local_files_only"] is False
    assert str(calls["map_location"]) == "cpu"
    assert state["param"].dtype == torch.float16
