from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch


CONFIG_FILE = "config.json"
WEIGHTS_FILE = "pytorch_model.bin"


def _resolve_directory(model_name: str | Path) -> Path:
    """Resolve ``model_name`` to a local directory path.

    Remote (Hub) resolution is intentionally not implemented in the scaffold to
    keep the dependency surface minimal. The helper nonetheless performs basic
    validation so callers receive actionable errors.
    """

    model_path = Path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"expected a directory path, received: {model_path}")
    return model_path


def load_config_hf(model_name: str | Path) -> dict:
    """Load a configuration dictionary from a local directory.

    Args:
        model_name: Hugging Face repo id or local directory. Only local
            directories are supported in the scaffold.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the directory or configuration file is missing.
        ValueError: If the configuration file cannot be parsed.
    """

    directory = _resolve_directory(model_name)
    config_path = directory / CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration JSON must decode to a dictionary")
    return data


def load_state_dict_hf(
    model_name: str | Path,
    device=None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Load a PyTorch state dict from a local directory.

    Args:
        model_name: Hugging Face repo id or local directory. Only local
            directories are supported in the scaffold.
        device: Optional device mapping passed to :func:`torch.load`.
        dtype: Optional dtype cast applied to each tensor.

    Returns:
        Mapping from parameter names to tensors.

    Raises:
        FileNotFoundError: If the directory or weights file is missing.
    """

    directory = _resolve_directory(model_name)
    weights_path = directory / WEIGHTS_FILE
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    if dtype is not None:
        state_dict = {name: tensor.to(dtype) for name, tensor in state_dict.items()}
    return state_dict


def save_pretrained_local(
    config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
    save_directory: str | Path,
) -> None:
    """Persist configuration and weights to ``save_directory``.

    The helper mirrors the Hugging Face ``save_pretrained`` contract but only
    implements the local path variant for the scaffold.

    Args:
        config: Serializable configuration mapping.
        state_dict: Model parameters to save via :func:`torch.save`.
        save_directory: Directory to create or overwrite files in.
    """

    path = Path(save_directory)
    path.mkdir(parents=True, exist_ok=True)

    config_path = path / CONFIG_FILE
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True)

    weights_path = path / WEIGHTS_FILE
    torch.save(dict(state_dict), weights_path)
