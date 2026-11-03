import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='session', autouse=True)
def native_build():
    from scripts.build import build
    return build()
