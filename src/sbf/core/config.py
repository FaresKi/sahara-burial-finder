"""Settings loaded from config/settings.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


class AOISettings(BaseModel):
    crs: str


class Settings(BaseModel):
    root: Path
    config_dir: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    outputs_public: Path
    outputs_restricted: Path
    aois: dict[str, AOISettings]
    tile_sizes_m: list[int] = [64, 128]
    overlap: float = 0.5
    modern_resolution_m: float = 10

    def crs(self, aoi_id: str) -> str:
        """Metric CRS (EPSG string) for an AOI."""
        try:
            return self.aois[aoi_id].crs
        except KeyError:
            raise KeyError(f"unknown AOI {aoi_id!r}; known: {sorted(self.aois)}") from None


def project_root() -> Path:
    """Project root: $SBF_ROOT if set, else the repository containing this package."""
    env = os.environ.get("SBF_ROOT")
    return Path(env).resolve() if env else _DEFAULT_ROOT


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load KEY=value lines from `<root>/.env` into os.environ without overriding existing vars.

    Returns the names that were set. Blank values are skipped; values are never logged.
    """
    path = path or project_root() / ".env"
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_settings(root: Path | str | None = None) -> Settings:
    """Load `<root>/config/settings.yaml`; relative paths resolve against root."""
    root = Path(root).resolve() if root is not None else project_root()
    raw = yaml.safe_load((root / "config" / "settings.yaml").read_text())
    p = raw["paths"]
    return Settings(
        root=root,
        config_dir=root / p.get("config_dir", "config"),
        data_raw=root / p["data_raw"],
        data_interim=root / p["data_interim"],
        data_processed=root / p["data_processed"],
        outputs_public=root / p["outputs_public"],
        outputs_restricted=root / p["outputs_restricted"],
        aois=raw["aois"],
        tile_sizes_m=raw.get("tile_sizes_m", [64, 128]),
        overlap=raw.get("overlap", 0.5),
        modern_resolution_m=raw.get("modern_resolution_m", 10),
    )
