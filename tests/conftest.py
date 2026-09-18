from pathlib import Path

import pandas as pd
import pytest

from support_ticket_ai.config import Settings
from support_ticket_ai.dataset import Dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset():
    return Dataset(ROOT / "data/support_tickets.csv")


@pytest.fixture
def as_of():
    return pd.Timestamp("2024-03-30T18:06:00Z")


@pytest.fixture
def settings():
    return Settings(data_path=ROOT / "data/support_tickets.csv", groq_api_key=None, _env_file=None)
