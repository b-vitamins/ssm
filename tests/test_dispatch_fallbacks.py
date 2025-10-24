from ssm.utils import dispatch


def test_backend_selection_default_python():
    # With no compiled extensions in this scaffold, we should see 'python'
    backend = dispatch.get_available_backend()
    assert backend == "python"


def test_has_kernels_false_in_scaffold():
    assert dispatch.has_cpu_kernels() is False
    assert dispatch.has_cuda_kernels() is False
