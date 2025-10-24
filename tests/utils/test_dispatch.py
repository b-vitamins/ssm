import pytest

from ssm.utils import dispatch


def test_python_reference_available():
    assert dispatch.has_python_reference() is True


def test_backend_selection_prefers_available(monkeypatch):
    monkeypatch.setattr(dispatch, "has_cuda_kernels", lambda: False)
    monkeypatch.setattr(dispatch, "has_cpu_kernels", lambda: False)
    monkeypatch.setattr(dispatch, "has_python_reference", lambda: True)
    assert dispatch.get_available_backend() == "python"

    monkeypatch.setattr(dispatch, "has_cuda_kernels", lambda: True)
    assert dispatch.get_available_backend() == "cuda"


def test_backend_preference_and_error(monkeypatch):
    monkeypatch.setattr(dispatch, "has_cuda_kernels", lambda: False)
    monkeypatch.setattr(dispatch, "has_cpu_kernels", lambda: True)
    monkeypatch.setattr(dispatch, "has_python_reference", lambda: True)
    assert dispatch.get_available_backend("cpu") == "cpu"
    assert dispatch.get_available_backend("python") == "python"

    with pytest.raises(ValueError):
        dispatch.get_available_backend("tpu")

    monkeypatch.setattr(dispatch, "has_cpu_kernels", lambda: False)
    monkeypatch.setattr(dispatch, "has_python_reference", lambda: False)
    with pytest.raises(RuntimeError):
        dispatch.get_available_backend()
