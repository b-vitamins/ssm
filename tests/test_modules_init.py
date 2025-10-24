import inspect

from ssm.modules import Mamba1, Mamba2, Block, MHA, GatedMLP


def test_init_signatures_minimal():
    # Ensure constructors accept the documented core parameters
    assert "d_model" in inspect.signature(Mamba1).parameters
    assert "d_model" in inspect.signature(Mamba2).parameters
    assert "dim" in inspect.signature(Block).parameters
    assert "embed_dim" in inspect.signature(MHA).parameters
    assert "in_features" in inspect.signature(GatedMLP).parameters
