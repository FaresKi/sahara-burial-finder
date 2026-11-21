"""Windowed / decimated access to frames so nothing assumes the whole frame fits in memory.

`FrameSource` is the minimal interface used by matching (`read_decimated`) and warping
(`read_window`). `ArraySource` wraps an in-memory array (tests, small frames); `RasterSource`
reads a GeoTIFF lazily with rasterio (`out_shape` decimated reads use overviews when present).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from sbf.features.preprocess import clahe


@runtime_checkable
class FrameSource(Protocol):
    shape: tuple[int, int]
    dtype: np.dtype

    def read_window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Pixels rows [r0, r1) x cols [c0, c1), clipped to the frame."""
        ...

    def read_decimated(self, factor: float) -> np.ndarray:
        """Whole frame averaged down by `factor` (>= 1) as float32."""
        ...


class ArraySource:
    def __init__(self, array: np.ndarray):
        a = np.asarray(array)
        if a.ndim != 2:
            raise ValueError("frame must be a 2-D array")
        self.array = a
        self.shape = (int(a.shape[0]), int(a.shape[1]))
        self.dtype = a.dtype

    def read_window(self, r0, r1, c0, c1):
        return self.array[max(r0, 0) : r1, max(c0, 0) : c1]

    def read_decimated(self, factor):
        import cv2

        f = self.array.astype("float32")
        if factor <= 1.0:
            return f
        w = max(8, round(f.shape[1] / factor))
        h = max(8, round(f.shape[0] / factor))
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)


class RasterSource:
    """Lazy band-1 reader. Keeps no pixel data; opens the file per call."""

    def __init__(self, path: str | Path, band: int = 1):
        self.path = Path(path)
        self.band = band
        with _quiet(), rasterio.open(self.path) as src:
            self.shape = (src.height, src.width)
            self.dtype = np.dtype(src.dtypes[band - 1])

    def read_window(self, r0, r1, c0, c1):
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(r1), self.shape[0]), min(int(c1), self.shape[1])
        if r1 <= r0 or c1 <= c0:
            return np.zeros((0, 0), self.dtype)
        with _quiet(), rasterio.open(self.path) as src:
            return src.read(self.band, window=Window(c0, r0, c1 - c0, r1 - r0))

    def read_decimated(self, factor):
        h, w = self.shape
        f = max(1.0, float(factor))
        oh, ow = max(8, round(h / f)), max(8, round(w / f))
        if f <= 1.0:
            oh, ow = h, w
        with _quiet(), rasterio.open(self.path) as src:
            return src.read(self.band, out_shape=(oh, ow), resampling=Resampling.average).astype(
                "float32"
            )


class _quiet:
    def __enter__(self):
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", NotGeoreferencedWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def as_source(frame) -> FrameSource:
    if isinstance(frame, FrameSource):
        return frame
    if isinstance(frame, str | Path):
        return RasterSource(frame)
    return ArraySource(frame)


def downsample_source(src: FrameSource, factor: float) -> tuple[np.ndarray, float]:
    """Decimated read + CLAHE. Returns (small_unit_image, scale): full-res px = small px * scale."""
    small = src.read_decimated(max(1.0, float(factor)))
    scale = src.shape[1] / small.shape[1]
    return clahe(small), float(scale)
