from pathlib import Path

import pytest

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture
def sample_repo() -> Path:
    return SAMPLE_REPO
