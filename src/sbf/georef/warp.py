"""Step 6: warp a full-resolution frame (or strip) into a metric grid by inverse mapping.

The forward model (pixel -> xy) is inverted approximately on a lattice (`fit.invert_on_lattice`);
the inverse is evaluated on a coarse lattice per output block, linearly upsampled to per-pixel
source coordinates and applied with `cv2.remap`. Streaming: for every output block only the
needed source window is read (`FrameSource.read_window`), so memory is bounded by the block size,
not the frame size. `warp_to_file` also writes blocks straight to disk. Strips share `Grid`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from sbf.georef.fit import TransformModel, invert_on_lattice
from sbf.georef.frame_io import FrameSource, as_source
from sbf.georef.models import UTM_CRS

MAX_OUTPUT_PIXELS = 600_000_000


@dataclass
class WarpResult:
    array: np.ndarray
    transform: Affine
    crs: str
    nodata: float
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    def window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        return self.array[max(r0, 0) : r1, max(c0, 0) : c1]


def estimate_resolution_m(model: TransformModel, shape: tuple[int, int]) -> float:
    """Ground size of one frame pixel at the frame centre (mean of x and y steps), 2 decimals."""
    h, w = shape[:2]
    c = np.array([[w / 2, h / 2], [w / 2 + 50, h / 2], [w / 2, h / 2 + 50]])
    p = model(c)
    r = 0.5 * (np.hypot(*(p[1] - p[0])) + np.hypot(*(p[2] - p[0]))) / 50.0
    return round(float(r), 2)


def frame_bounds_xy(
    model: TransformModel,
    shape: tuple[int, int],
    n: int = 50,
    extent: tuple[float, float, float, float] | None = None,
):
    """(xmin, ymin, xmax, ymax) of the model-mapped, densified frame border (or of `extent` =
    (px_x0, px_y0, px_x1, px_y1), e.g. one strip)."""
    h, w = shape[:2]
    x0, y0, x1, y1 = extent if extent is not None else (0.0, 0.0, float(w), float(h))
    t = np.linspace(0, 1, n)
    edge = np.concatenate(
        [
            np.column_stack([x0 + t * (x1 - x0), np.full(n, y0)]),
            np.column_stack([x0 + t * (x1 - x0), np.full(n, y1)]),
            np.column_stack([np.full(n, x0), y0 + t * (y1 - y0)]),
            np.column_stack([np.full(n, x1), y0 + t * (y1 - y0)]),
        ]
    )
    p = model(edge)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def _upsample(coarse: np.ndarray, pos_r: np.ndarray, pos_c: np.ndarray, nr: int, nc: int):
    """Separable linear interpolation of a coarse lattice (rows at pos_r, cols at pos_c)."""
    cols = np.arange(nc)
    rows = np.arange(nr)
    tmp = np.stack([np.interp(cols, pos_c, coarse[i]) for i in range(coarse.shape[0])])
    return np.stack([np.interp(rows, pos_r, tmp[:, j]) for j in range(nc)], axis=1)


@dataclass(frozen=True)
class Grid:
    """North-up output grid snapped to multiples of `res`."""

    x0: float
    y1: float
    res: float
    W: int
    H: int

    @classmethod
    def from_bounds(cls, bounds, res: float) -> Grid:
        xmin, ymin, xmax, ymax = bounds
        x0 = math.floor(xmin / res) * res
        y1 = math.ceil(ymax / res) * res
        W = math.ceil((xmax - x0) / res)
        H = math.ceil((y1 - ymin) / res)
        if W * H > MAX_OUTPUT_PIXELS:
            raise ValueError(
                f"output {W}x{H} px exceeds {MAX_OUTPUT_PIXELS}; check resolution/model"
            )
        return cls(x0, y1, res, W, H)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y1)

    def blocks(self, block: int) -> Iterator[tuple[int, int, int, int]]:
        for r0 in range(0, self.H, block):
            for c0 in range(0, self.W, block):
                yield r0, c0, min(block, self.H - r0), min(block, self.W - c0)


def block_source_coords(inv, grid: Grid, r0: int, c0: int, nr: int, nc: int, step: int = 16):
    """Per-pixel source coordinates (corner convention) of one output block: (px_x, px_y)."""
    pos_r = np.unique(np.append(np.arange(0, nr, step), nr - 1)).astype(float)
    pos_c = np.unique(np.append(np.arange(0, nc, step), nc - 1)).astype(float)
    gx = grid.x0 + (c0 + pos_c + 0.5) * grid.res
    gy = grid.y1 - (r0 + pos_r + 0.5) * grid.res
    X, Y = np.meshgrid(gx, gy)
    px = inv(np.column_stack([X.ravel(), Y.ravel()])).reshape(len(pos_r), len(pos_c), 2)
    sx = _upsample(px[..., 0], pos_r, pos_c, nr, nc)
    sy = _upsample(px[..., 1], pos_r, pos_c, nr, nc)
    return sx, sy


def sample_source(
    source: FrameSource,
    sx: np.ndarray,
    sy: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample at corner-convention px (sx, sy), reading only the needed window.

    Returns (tile in source dtype, inside mask). `valid` further restricts the inside mask.
    """
    h, w = source.shape
    inside = (sx >= 0) & (sx <= w) & (sy >= 0) & (sy <= h)
    if valid is not None:
        inside &= valid
    tile = np.zeros(sx.shape, dtype=source.dtype)
    if not inside.any():
        return tile, inside
    mx, my = sx - 0.5, sy - 0.5  # corner -> cv2 centre convention
    c0 = max(math.floor(mx[inside].min()) - 2, 0)
    c1 = min(math.ceil(mx[inside].max()) + 3, w)
    r0 = max(math.floor(my[inside].min()) - 2, 0)
    r1 = min(math.ceil(my[inside].max()) + 3, h)
    win = source.read_window(r0, r1, c0, c1)
    if win.size == 0:
        inside[:] = False
        return tile, inside
    if win.dtype not in (np.uint8, np.uint16, np.float32):
        win = win.astype("float32")
    out = cv2.remap(
        win,
        (mx - c0).astype("float32"),
        (my - r0).astype("float32"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if source.dtype != out.dtype:
        out = out.astype(source.dtype)
    return out, inside


def _nodata_for(dtype, nodata):
    if nodata is not None:
        return nodata
    return 0 if np.issubdtype(np.dtype(dtype), np.integer) else float("nan")


def _prepare(source, model, resolution_m, bounds, extent=None):
    src = as_source(source)
    res = float(resolution_m or estimate_resolution_m(model, src.shape))
    b = bounds or frame_bounds_xy(model, src.shape, extent=extent)
    return src, res, Grid.from_bounds(b, res)


def _warp_blocks(src, model, grid, block, step):
    inv = invert_on_lattice(model, src.shape)
    for r0, c0, nr, nc in grid.blocks(block):
        sx, sy = block_source_coords(inv, grid, r0, c0, nr, nc, step)
        tile, inside = sample_source(src, sx, sy)
        yield r0, c0, tile, inside


def warp_frame(
    src,
    model: TransformModel,
    *,
    resolution_m: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    crs: str = UTM_CRS,
    block: int = 1024,
    step: int = 16,
    nodata: float | None = None,
) -> WarpResult:
    """Resample `src` (2-D array or `FrameSource`) into a north-up grid of `resolution_m` metres.

    `bounds` (xmin, ymin, xmax, ymax) defaults to the mapped frame extent, snapped outward to
    multiples of the resolution. Pixels outside the frame get `nodata` (0 for integer dtypes,
    NaN for floats unless given). Source pixels are read per output block (windowed).
    """
    if isinstance(src, np.ndarray) and src.ndim != 2:
        raise ValueError("warp_frame expects a 2-D array (warp each band separately)")
    source, res, grid = _prepare(src, model, resolution_m, bounds)
    nodata = _nodata_for(source.dtype, nodata)
    out = np.full((grid.H, grid.W), nodata, dtype=source.dtype)
    for r0, c0, tile, inside in _warp_blocks(source, model, grid, block, step):
        sub = out[r0 : r0 + tile.shape[0], c0 : c0 + tile.shape[1]]
        sub[inside] = tile[inside]
    return WarpResult(out, grid.transform, crs, float(nodata), res)


@dataclass
class WarpFile:
    path: Path
    transform: Affine
    crs: str
    nodata: float
    resolution_m: float
    shape: tuple[int, int]

    def window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Windowed read of the written file (overlay chips never load the whole output)."""
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, self.shape[0]), min(c1, self.shape[1])
        with rasterio.open(self.path) as s:
            return s.read(1, window=Window(c0, r0, max(c1 - c0, 0), max(r1 - r0, 0)))


def open_tiled_output(path: Path, grid: Grid, dtype, nodata, crs: str | None = UTM_CRS):
    """Tiled deflate GeoTIFF writer; `crs=None` writes no georeference (pre-georef assembly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    geo = {"crs": crs, "transform": grid.transform} if crs else {}
    return rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.H,
        width=grid.W,
        count=1,
        dtype=str(np.dtype(dtype)),
        **geo,
        nodata=None if (isinstance(nodata, float) and math.isnan(nodata)) else nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )


def warp_to_file(
    src,
    model: TransformModel,
    path: str | Path,
    *,
    resolution_m: float | None = None,
    bounds=None,
    crs: str = UTM_CRS,
    block: int = 1024,
    step: int = 16,
    nodata: float | None = None,
) -> WarpFile:
    """Like `warp_frame` but streams each block to a tiled GeoTIFF (no whole-output array)."""
    source, res, grid = _prepare(src, model, resolution_m, bounds)
    nd = _nodata_for(source.dtype, nodata)
    path = Path(path)
    with open_tiled_output(path, grid, source.dtype, nd, crs) as dst:
        for r0, c0, tile, inside in _warp_blocks(source, model, grid, block, step):
            blk = np.where(inside, tile, np.asarray(nd, dtype=tile.dtype)).astype(source.dtype)
            dst.write(blk, 1, window=Window(c0, r0, tile.shape[1], tile.shape[0]))
    return WarpFile(path, grid.transform, crs, float(nd), res, (grid.H, grid.W))
