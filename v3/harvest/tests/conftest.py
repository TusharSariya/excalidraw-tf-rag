from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def s2_paper_forceatlas2() -> dict:
    import json

    return json.loads((FIXTURES / "s2_paper_forceatlas2.json").read_text(encoding="utf-8"))
