import inspect

import ssm
from ssm.modules import Mamba1, Mamba2, Block, MHA, GatedMLP
from ssm.models import MambaConfig, MambaLMHeadModel
from ssm.ops import (
    selective_scan,
    selective_state_step,
    ssd_chunk_scan,
    dw_causal_conv,
    fused_layer_norm,
)


def test_version_str():
    assert isinstance(ssm.__version__, str)


def _has_google_sections(fn):
    doc = inspect.getdoc(fn) or ""
    # Simple heuristics to enforce presence of sections in docstrings
    must_have = ["Args:", "Returns:"]
    return all(x in doc for x in must_have)


def test_ops_signatures_and_docstrings():
    for fn in [
        selective_scan,
        selective_state_step,
        ssd_chunk_scan,
        dw_causal_conv,
        fused_layer_norm,
    ]:
        sig = inspect.signature(fn)
        assert len(sig.parameters) >= 1
        assert _has_google_sections(fn)


def test_module_class_docstrings():
    for cls in [Mamba1, Mamba2, Block, MHA, GatedMLP, MambaLMHeadModel]:
        doc = inspect.getdoc(cls) or ""
        assert "Args:" in doc or "Attributes:" in doc


def test_model_config_dataclass_defaults():
    cfg = MambaConfig()
    assert cfg.d_model > 0 and cfg.n_layer > 0 and cfg.vocab_size > 0
    assert isinstance(cfg.ssm_cfg, dict)
    assert isinstance(cfg.attn_cfg, dict)
