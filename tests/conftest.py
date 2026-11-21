import shutil
from pathlib import Path

import pytest

from sbf.core.config import Settings, load_settings

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_project(tmp_path, monkeypatch) -> Settings:
    """A temp project root with a copy of config/ and settings pointing there.

    Also sets SBF_ROOT so code calling `load_settings()` without args uses it.
    """
    shutil.copytree(REPO / "config", tmp_path / "config")
    for d in ("data/raw", "data/interim", "data/processed", "outputs/public", "outputs/restricted"):
        (tmp_path / d).mkdir(parents=True)
    monkeypatch.setenv("SBF_ROOT", str(tmp_path))
    return load_settings(tmp_path)
