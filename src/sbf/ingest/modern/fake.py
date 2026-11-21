"""Synthetic exporter for tests and offline dry runs. Never touches the network."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from sbf.core.provenance import write_prov
from sbf.ingest.modern.exporter import DatasetCheck, ExportReport, ProductResult
from sbf.ingest.modern.grid import Grid, grid_bounds
from sbf.ingest.modern.layers import NODATA_RAW, PRODUCTS


class FakeExporter:
    """Writes synthetic raw products in the same layout as the live exporter.

    Coarse products (PALSAR, DEM) are written at their native resolution on a grid
    offset from the common one so that co-registration is genuinely exercised.
    `fail` lists product keys to report as failed (dataset 'not found').
    """

    def __init__(self, fail: set[str] | None = None, native_coarse: bool = True) -> None:
        self.fail = fail or set()
        self.native_coarse = native_coarse

    def verify_datasets(self, grid: Grid, products: list[str] | None = None) -> list[DatasetCheck]:
        return [
            DatasetCheck(k, None, False, detail="fake: not found")
            if k in self.fail
            else DatasetCheck(k, PRODUCTS[k].dataset_candidates[0], True, 1)
            for k in products or list(PRODUCTS)
        ]

    @staticmethod
    def _values(key: str, band: str, shape: tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(f"{key}/{band}".encode()))
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = (xx / max(w, 1) + yy / max(h, 1)) / 2
        noise = rng.normal(0, 1, shape).astype(np.float32)
        if key.startswith("s2"):
            return 0.2 + 0.2 * base + 0.01 * noise
        if key == "s1":
            return (
                (-15 + 3 * base + 0.3 * noise)
                if band.endswith("mean")
                else 1.5 + 0.1 * np.abs(noise)
            )
        if key.startswith("palsar"):
            return 2000 + 1500 * base + 50 * noise
        # dem: tilted plane + central bump
        bump = 40 * np.exp(-(((xx - w / 2) / (w / 6)) ** 2 + ((yy - h / 2) / (h / 6)) ** 2))
        return 300 + 50 * base + bump + 0.2 * noise

    def export_aoi(
        self,
        aoi_id: str,
        out_dir: Path,
        *,
        grid: Grid,
        products: list[str] | None = None,
    ) -> ExportReport:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        transform, _, _, crs = grid
        left, bottom, right, top = grid_bounds(grid)
        report = ExportReport(aoi_id, {"start": "fake", "end": "fake"})
        for key in products or list(PRODUCTS):
            spec = PRODUCTS[key]
            if key in self.fail:
                report.results.append(
                    ProductResult(key, "failed", None, 0, None, "fake: dataset not found")
                )
                continue
            res = spec.native_resolution_m if self.native_coarse else transform.a
            if res > transform.a:
                left_, top_ = left - res / 2, top + res / 2  # deliberately misaligned
                w = int(np.ceil((right - left_) / res)) + 1
                h = int(np.ceil((top_ - bottom) / res)) + 1
            else:
                left_, top_, res = left, top, transform.a
                w, h = grid[1], grid[2]
            tr = Affine(res, 0, left_, 0, -res, top_)
            path = out_dir / f"{key}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=w,
                height=h,
                count=len(spec.bands),
                dtype="float32",
                crs=crs,
                transform=tr,
                nodata=NODATA_RAW,
            ) as dst:
                for i, b in enumerate(spec.bands, 1):
                    arr = self._values(key, b, (h, w)).astype(np.float32)
                    dst.write(arr, i)
                    dst.set_band_description(i, b)
            write_prov(
                path,
                source="FAKE synthetic exporter",
                source_id=spec.dataset_candidates[0],
                acquisition_date=None,
                steps=["synthetic data for tests; not real imagery"],
            )
            report.results.append(
                ProductResult(key, "ok", spec.dataset_candidates[0], 1, str(path))
            )
        report.write(out_dir)
        return report
