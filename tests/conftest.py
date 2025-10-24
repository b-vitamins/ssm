import sys
from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-goldens",
        action="store_true",
        default=False,
        help="Run golden tests that compare against stored outputs.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "golden: marks tests as golden (skip by default)"
    )
    # Ensure src/ is importable without installing the package
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@pytest.fixture
def run_goldens(request):
    return request.config.getoption("--run-goldens")
