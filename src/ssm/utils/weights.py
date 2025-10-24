from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, cast

import torch


CONFIG_FILE = "config.json"
WEIGHTS_FILE = "pytorch_model.bin"


def _resolve_checkpoint_file(
    model_name: str | Path,
    filename: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return the path to ``filename`` for a given checkpoint identifier.

    The helper first checks for local directories or files before deferring to
    :func:`huggingface_hub.cached_file` for remote Hub entries. ``cache_dir``
    mirrors the Hugging Face API and is stringified to avoid ``Path`` handling
    issues inside the library.

    Args:
        model_name: Local directory / file path or Hugging Face repo id.
        filename: Target file to locate (for example ``config.json``).
        revision: Optional Git revision used for Hub lookups.
        token: Optional authentication token forwarded to the Hub client.
        cache_dir: Optional cache directory override for Hub downloads.
        local_files_only: If ``True``, rely exclusively on local cache entries.

    Returns:
        Path pointing to ``filename`` on disk.

    Raises:
        FileNotFoundError: When neither a local file nor a cached Hub entry
            exists for ``model_name``.
        RuntimeError: If the Hugging Face Hub client is unavailable when a
            remote resolution is required.
    """

    model_path = Path(model_name).expanduser()
    is_path_like = not isinstance(model_name, str)

    if is_path_like and not model_path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {model_path}")

    if model_path.is_file():
        if model_path.name != filename:
            raise ValueError(f"expected '{filename}' but received file: {model_path}")
        return model_path

    if model_path.is_dir():
        candidate = model_path / filename
        if not candidate.exists():
            raise FileNotFoundError(f"missing checkpoint file: {candidate}")
        return candidate

    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    try:
        from huggingface_hub import (  # pyright: ignore[reportMissingImports]
            cached_file,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub is required to download remote checkpoints"
        ) from exc

    cached_file_fn = cast(Callable[..., str | None], cached_file)

    resolved_file = cached_file_fn(
        str(model_name),
        filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir_str,
        local_files_only=local_files_only,
        _raise_exceptions_for_missing_entries=False,
    )
    if resolved_file is None:
        raise FileNotFoundError(
            f"unable to resolve '{filename}' for checkpoint '{model_name}'"
        )
    return Path(resolved_file)


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

    config_path = _resolve_checkpoint_file(
        model_name,
        CONFIG_FILE,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
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

    weights_path = _resolve_checkpoint_file(
        model_name,
        WEIGHTS_FILE,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # Loading directly to the target device when ``dtype`` changes can trigger
    # unnecessary peak memory usage, so mirror the upstream behaviour by
    # remapping to CPU before casting.
    mapped_device = device
    if dtype is not None and dtype is not torch.float32:
        mapped_device = torch.device("cpu")

    state_dict = torch.load(weights_path, map_location=mapped_device)

    if dtype is not None:
        state_dict = {
            name: tensor.to(dtype=dtype) for name, tensor in state_dict.items()
        }

    if device is not None and (
        mapped_device is None or str(mapped_device) != str(device)
    ):
        state_dict = {
            name: tensor.to(device=device) for name, tensor in state_dict.items()
        }

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
