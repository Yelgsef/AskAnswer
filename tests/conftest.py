"""Shared fixtures. Tests that need the released data (data/, not in git) are skipped when it is absent."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "train"


@pytest.fixture(scope="session")
def train_answers() -> list[dict]:
    if not (TRAIN / "answers.jsonl").exists():
        pytest.skip("data/train is not downloaded")
    with open(TRAIN / "answers.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
