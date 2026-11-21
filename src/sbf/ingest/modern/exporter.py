"""GEE export interface and the Earth Engine implementation (live use is a human gate).

Design choice: direct download via `Image.getDownloadURL`, tiled to stay under the ~32 MB
request limit, rather than `Export.image.toDrive`.
Why: it is synchronous and scriptable (no Drive hand-off or manual download, so
`sbf ingest modern export` is reproducible end to end), it needs no Drive/GCS scopes or
billing, and every tile is requested on the final 10 m UTM grid (`crs_transform`), so tiles
land pixel-aligned and are written straight into one GeoTIFF window by window. The cost is
many small requests (REG is a few hundred tiles in total); each is retried.

Dataset IDs are NOT assumed to exist: every product is verified at runtime (asset exists,
filtered collection non-empty over the AOI, required bands present). Failures are recorded
in `export_report.json` and the product is skipped; nothing is substituted or synthesised.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.windows import Window

from sbf.core.provenance import write_prov
from sbf.ingest.modern.grid import Grid, grid_bounds, grid_profile
from sbf.ingest.modern.layers import NODATA_RAW, PRODUCTS, S2_BANDS, S2_SEASONS

REPORT_NAME = "export_report.json"


class GeeNotConfiguredError(RuntimeError):
    """GEE_PROJECT missing or Earth Engine not authenticated/initialisable."""


@dataclass
class DatasetCheck:
    product: str
    dataset_id: str | None
    ok: bool
    n_images: int = 0
    bands: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ProductResult:
    product: str
    status: str  # ok | failed
    dataset_id: str | None = None
    n_images: int = 0
    path: str | None = None
    error: str | None = None


@dataclass
class ExportReport:
    aoi_id: str
    window: dict
    results: list[ProductResult] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / REPORT_NAME
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @property
    def failed(self) -> list[ProductResult]:
        return [r for r in self.results if r.status != "ok"]


@runtime_checkable
class GeeExporter(Protocol):
    """Anything that can verify GEE datasets and export raw AOI products as GeoTIFFs."""

    def verify_datasets(
        self, grid: Grid, products: list[str] | None = None
    ) -> list[DatasetCheck]: ...

    def export_aoi(
        self,
        aoi_id: str,
        out_dir: Path,
        *,
        grid: Grid,
        products: list[str] | None = None,
    ) -> ExportReport: ...


# --------------------------------------------------------------------------- tiling (pure)


def plan_tiles(
    width: int, height: int, n_bands: int, *, bytes_per_px: int = 4, max_bytes: int = 20_000_000
) -> list[tuple[int, int, int, int]]:
    """Split a raster into (col_off, row_off, w, h) tiles under `max_bytes` uncompressed."""
    side = max(1, int(math.sqrt(max_bytes / (n_bands * bytes_per_px))))
    tiles = []
    for r in range(0, height, side):
        for c in range(0, width, side):
            tiles.append((c, r, min(side, width - c), min(side, height - r)))
    return tiles


def _read_tile_bytes(blob: bytes, n_bands: int) -> np.ndarray:
    """Decode a getDownloadURL payload (GeoTIFF or zip of single-band GeoTIFFs) to (n, h, w)."""
    if blob[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")))
            arrs = []
            for n in names:
                with rasterio.MemoryFile(zf.read(n)) as mf, mf.open() as ds:
                    arrs.append(ds.read())
        arr = np.concatenate(arrs, axis=0)
    else:
        with rasterio.MemoryFile(blob) as mf, mf.open() as ds:
            arr = ds.read()
    if arr.shape[0] != n_bands:
        raise ValueError(f"tile has {arr.shape[0]} bands, expected {n_bands}")
    return arr.astype(np.float32)


def write_tiles(
    out_path: Path,
    grid: Grid,
    band_names: list[str],
    tiles: list[tuple[int, int, int, int]],
    get_tile: Callable[[tuple[int, int, int, int]], np.ndarray],
) -> None:
    """Assemble tiles (each (n,h,w) from `get_tile`) into one tiled, compressed GeoTIFF."""
    prof = grid_profile(grid)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        count=len(band_names),
        dtype="float32",
        nodata=NODATA_RAW,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        **prof,
    ) as dst:
        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)
        for t in tiles:
            c, r, w, h = t
            arr = get_tile(t)
            if arr.shape[1:] != (h, w):
                raise ValueError(f"tile {t}: got shape {arr.shape[1:]}, expected {(h, w)}")
            dst.write(arr, window=Window(c, r, w, h))


# --------------------------------------------------------------------------- Earth Engine


def _default_fetch(url: str) -> bytes:
    with urlopen(url, timeout=600) as resp:
        return resp.read()


class EarthEngineExporter:
    """Live exporter. `ee` is imported lazily; nothing runs until `export_aoi`/`verify_datasets`."""

    REQUIRED_BANDS: ClassVar[dict[str, list[str]]] = {
        "s2": ["SCL", *S2_BANDS],
        "s1": ["VV", "VH"],
        "palsar": ["HH", "HV"],
        "dem": ["DEM"],
    }

    def __init__(
        self,
        project: str | None = None,
        *,
        years: int = 2,
        end_date: date | None = None,
        max_tile_bytes: int = 20_000_000,
        retries: int = 3,
        ee_module: Any = None,
        fetch: Callable[[str], bytes] = _default_fetch,
    ) -> None:
        self.project = project or os.environ.get("GEE_PROJECT")
        self.years = years
        self.end_date = end_date or datetime.now(UTC).date()
        self.max_tile_bytes = max_tile_bytes
        self.retries = retries
        self._ee = ee_module
        self._fetch = fetch
        self._initialised = ee_module is not None

    # -- setup
    def _ee_mod(self) -> Any:
        if self._ee is None:
            import ee  # lazy: heavy optional dependency

            self._ee = ee
        if not self._initialised:
            if not self.project:
                raise GeeNotConfiguredError(
                    "GEE_PROJECT is not set. Export it (a Cloud project registered for Earth "
                    "Engine) and run `earthengine authenticate` first; see notes/PHASE_2_NOTES.md."
                )
            try:
                self._ee.Initialize(project=self.project)
            except Exception as e:  # EEException subclasses vary by version
                raise GeeNotConfiguredError(
                    f"Earth Engine initialisation failed ({e}). Run `earthengine authenticate`."
                ) from e
            self._initialised = True
        return self._ee

    @property
    def window(self) -> dict:
        start = self.end_date - timedelta(days=365 * self.years)
        return {"start": start.isoformat(), "end": self.end_date.isoformat()}

    # -- collections
    @staticmethod
    def _family(product: str) -> str:
        return product.split("_")[0]

    def _region(self, ee: Any, grid: Grid) -> Any:
        left, bottom, right, top = grid_bounds(grid)
        return ee.Geometry.Rectangle([left, bottom, right, top], proj=grid[3], geodesic=False)

    def _collection(self, ee: Any, product: str, dataset_id: str, region: Any) -> Any:
        fam = self._family(product)
        col = ee.ImageCollection(dataset_id).filterBounds(region)
        w = self.window
        if fam in ("s2", "s1"):
            col = col.filterDate(w["start"], w["end"])
        if fam == "s2":
            suffix = product.removeprefix("s2_")
            if suffix in S2_SEASONS:
                months = S2_SEASONS[suffix]
                col = col.filter(
                    ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months])
                )
        elif fam == "s1":
            col = (
                col.filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            )
        return col

    # -- verification
    def _check_one(self, ee: Any, product: str, dataset_id: str, region: Any) -> DatasetCheck:
        chk = DatasetCheck(product, dataset_id, ok=False)
        try:
            ee.data.getAsset(dataset_id)
        except Exception as e:  # noqa: BLE001
            chk.detail = f"asset lookup failed: {e}"
            return chk
        try:
            col = self._collection(ee, product, dataset_id, region)
            chk.n_images = int(col.size().getInfo())
            if chk.n_images == 0:
                chk.detail = "no images over AOI/time window"
                return chk
            chk.bands = list(col.first().bandNames().getInfo())
        except Exception as e:  # noqa: BLE001
            chk.detail = f"collection query failed: {e}"
            return chk
        need = self.REQUIRED_BANDS[self._family(product)]
        lacking = [b for b in need if b not in chk.bands]
        if lacking:
            chk.detail = f"missing bands {lacking}; has {chk.bands}"
            return chk
        chk.ok = True
        return chk

    def verify_datasets(self, grid: Grid, products: list[str] | None = None) -> list[DatasetCheck]:
        """One check per product: the first candidate dataset ID that verifies, else the failure."""
        ee = self._ee_mod()
        region = self._region(ee, grid)
        out = []
        for key in products or list(PRODUCTS):
            tried = [
                self._check_one(ee, key, ds, region) for ds in PRODUCTS[key].dataset_candidates
            ]
            good = next((c for c in tried if c.ok), None)
            if good:
                out.append(good)
            else:
                detail = "; ".join(f"{c.dataset_id}: {c.detail}" for c in tried)
                out.append(DatasetCheck(key, None, False, detail=detail))
        return out

    # -- server-side image construction
    def _build_image(
        self, ee: Any, check: DatasetCheck, grid: Grid
    ) -> tuple[Any, list[str], list[str]]:
        """Return (image, output band names, extra provenance steps)."""
        key, ds = check.product, check.dataset_id
        region = self._region(ee, grid)
        col = self._collection(ee, key, ds, region)
        fam = self._family(key)
        steps = [f"GEE collection {ds}, {check.n_images} images, window {self.window}"]
        spec = PRODUCTS[key]

        if fam == "s2":
            has_qa = "QA60" in check.bands

            def mask(img: Any) -> Any:
                img = img.resample("bilinear")
                scl = img.select("SCL")
                bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
                if has_qa:
                    qa = img.select("QA60")
                    bad = bad.Or(qa.bitwiseAnd(1 << 10).neq(0)).Or(qa.bitwiseAnd(1 << 11).neq(0))
                return img.select(S2_BANDS).updateMask(bad.Not()).divide(10000)

            image = col.map(mask).median()
            steps.append("mask SCL in {3,8,9,10,11}" + (" + QA60 cloud bits" if has_qa else ""))
            steps.append("scale /10000, per-pixel median")
        elif fam == "s1":
            col = col.select(["VV", "VH"])
            red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
            image = col.reduce(red).select(
                ["VV_mean", "VV_stdDev", "VH_mean", "VH_stdDev"], list(spec.bands)
            )
            steps.append("IW, VV+VH; per-pixel mean and std of GEE-provided dB values")
        elif fam == "palsar":
            col = col.select(["HH", "HV"]).map(lambda i: i.resample("bilinear"))
            if key == "palsar_latest":
                first = col.sort("system:time_start", False).first()
                image = first
                try:
                    year = ee.Date(first.get("system:time_start")).format("YYYY").getInfo()
                    steps.append(f"most recent epoch/year: {year}")
                except Exception:  # noqa: BLE001
                    steps.append("most recent epoch (year lookup failed)")
            else:
                image = col.mean()
                steps.append("mean of DN over all available years")
        else:  # dem
            col = col.select("DEM").map(lambda i: i.resample("bilinear"))
            proj = col.first().projection()
            image = col.mosaic().setDefaultProjection(proj)
            steps.append("mosaic of GLO-30 tiles, bilinear resample")
        image = image.clip(region).toFloat().unmask(NODATA_RAW)
        return image, list(spec.bands), steps

    # -- download
    def _download(self, ee: Any, image: Any, bands: list[str], grid: Grid, out_path: Path) -> int:
        transform, width, height, crs = grid
        res = transform.a
        tiles = plan_tiles(width, height, len(bands), max_bytes=self.max_tile_bytes)

        def get_tile(t: tuple[int, int, int, int]) -> np.ndarray:
            c, r, w, h = t
            x0 = transform.c + c * res
            y1 = transform.f - r * res
            rect = ee.Geometry.Rectangle(
                [x0, y1 - h * res, x0 + w * res, y1], proj=crs, geodesic=False
            )
            params = {
                "bands": bands,
                "region": rect,
                "crs": crs,
                "crs_transform": [res, 0, x0, 0, -res, y1],
                "format": "GEO_TIFF",
            }
            last: Exception | None = None
            for attempt in range(self.retries):
                try:
                    url = image.select(bands).getDownloadURL(params)
                    return _read_tile_bytes(self._fetch(url), len(bands))
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(2**attempt)
            raise RuntimeError(f"tile {t} failed after {self.retries} attempts: {last}")

        write_tiles(out_path, grid, bands, tiles, get_tile)
        return len(tiles)

    def export_aoi(
        self,
        aoi_id: str,
        out_dir: Path,
        *,
        grid: Grid,
        products: list[str] | None = None,
    ) -> ExportReport:
        ee = self._ee_mod()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = ExportReport(aoi_id, self.window)
        for chk in self.verify_datasets(grid, products):
            if not chk.ok:
                report.results.append(
                    ProductResult(chk.product, "failed", None, chk.n_images, None, chk.detail)
                )
                continue
            final = out_dir / f"{chk.product}.tif"
            part = out_dir / f"{chk.product}.tif.part"
            try:
                image, bands, steps = self._build_image(ee, chk, grid)
                n = self._download(ee, image, bands, grid, part)
                part.replace(final)
                write_prov(
                    final,
                    source="Google Earth Engine",
                    source_id=chk.dataset_id or chk.product,
                    acquisition_date=None,
                    steps=[
                        *steps,
                        (
                            f"export via getDownloadURL, {n} tiles, {grid[0].a:g} m, "
                            f"{grid[3]}, nodata {NODATA_RAW:g}"
                        ),
                    ],
                )
                report.results.append(
                    ProductResult(chk.product, "ok", chk.dataset_id, chk.n_images, str(final))
                )
            except Exception as e:  # noqa: BLE001
                part.unlink(missing_ok=True)
                report.results.append(
                    ProductResult(
                        chk.product,
                        "failed",
                        chk.dataset_id,
                        chk.n_images,
                        None,
                        f"{type(e).__name__}: {e}",
                    )
                )
        report.write(out_dir)
        return report
