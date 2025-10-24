from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping

import torch


CONFIG_FILE = "config.json"
WEIGHTS_FILE = "pytorch_model.bin"


def _resolve_directory(
    model_name: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve ``model_name`` to a checkpoint directory.

    Local paths are returned directly. Remote Hugging Face Hub identifiers are
    downloaded with :func:`huggingface_hub.snapshot_download` when available.
    """

    model_path = Path(model_name)
    is_path_like = not isinstance(model_name, str)
    if is_path_like or model_path.is_absolute():
        if not model_path.exists():
            raise FileNotFoundError(f"model directory not found: {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"expected a directory path, received: {model_path}")
        return model_path
    if model_path.exists():
        if not model_path.is_dir():
            raise ValueError(f"expected a directory path, received: {model_path}")
        return model_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub is required to download remote checkpoints"
        ) from exc

    download_path = snapshot_download(
        repo_id=str(model_name),
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    return Path(download_path)


def load_config_hf(
    model_name: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict:
    """Load a configuration dictionary from a checkpoint directory.

    Args:
        model_name: Hugging Face repo id or local directory.
        revision: Optional Git revision to download from the Hub.
        token: Optional authentication token passed to the Hub client.
        cache_dir: Optional cache directory override for Hub downloads.
        local_files_only: If ``True``, only use local cache entries without
            attempting to reach the network.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the directory or configuration file is missing.
        ValueError: If the configuration file cannot be parsed.
    """

    directory = _resolve_directory(
        model_name,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
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
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a PyTorch state dict from a checkpoint directory.

    Args:
        model_name: Hugging Face repo id or local directory.
        device: Optional device mapping passed to :func:`torch.load`.
        dtype: Optional dtype cast applied to each tensor.
        revision: Optional Git revision to download from the Hub.
        token: Optional authentication token passed to the Hub client.
        cache_dir: Optional cache directory override for Hub downloads.
        local_files_only: If ``True``, only use local cache entries without
            attempting to reach the network.

    Returns:
        Mapping from parameter names to tensors.

    Raises:
        FileNotFoundError: If the directory or weights file is missing.
    """

    directory = _resolve_directory(
        model_name,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
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
