from __future__ import annotations


def load_config_hf(model_name: str) -> dict:
    """Load a HF config JSON (design-only stub).

    Args:
        model_name: Hugging Face repo id or local directory.

    Returns:
        Dict configuration.

    Raises:
        NotImplementedError: Implementation to be supplied later.
    """
    raise NotImplementedError("utils.load_config_hf is not implemented in the scaffold.")


def load_state_dict_hf(model_name: str, device=None, dtype=None) -> dict:
    """Load a HF PyTorch state dict (design-only stub).

    Args:
        model_name: Hugging Face repo id or local directory.
        device: Optional device mapping.
        dtype: Optional dtype cast.

    Returns:
        PyTorch state dict mapping parameter names to Tensors.

    Raises:
        NotImplementedError: Implementation to be supplied later.
    """
    raise NotImplementedError("utils.load_state_dict_hf is not implemented in the scaffold.")
