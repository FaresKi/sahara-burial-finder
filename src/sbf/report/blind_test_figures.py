"""Blind-test figures, leakage-safe by construction (SPEC section 0 rule 1).

Public figures only ever show: (a) recall@K curves vs random, (b) rank histograms of positive
tiles, (c) per known reference feature an imagery chip (footprint + margin) next to a score
panel where only tiles intersecting the footprint carry a colour and everything else is a
neutral mask colour. There is NO AOI-wide score heatmap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sbf.core.config import Settings, load_settings

log = logging.getLogger(__name__)
MASK_RGB = (200, 200, 200)  # neutral mask colour (uint8)
MARGIN_M = 100.0
MAX_K = 200


def recall_curve(best_ranks, n_tiles: int, tiles_per_feature, ks=None):
    """(ks, observed feature recall@K, random expectation) from positive-tile ranks only."""
    from sbf.detect.harness import feature_recall_at_k, random_feature_recall_at_k

    ks = np.arange(1, MAX_K + 1) if ks is None else np.asarray(ks)
    obs = [feature_recall_at_k(best_ranks, int(k)) for k in ks]
    rnd = [random_feature_recall_at_k(n_tiles, tiles_per_feature, int(k)) for k in ks]
    return ks, np.array(obs), np.array(rnd)


def score_panel(
    footprint,
    tile_boxes: dict[str, tuple[float, float, float, float]],
    tile_scores: dict[str, float],
    bounds: tuple[float, float, float, float],
    px_m: float,
) -> np.ndarray:
    """RGB uint8 panel over ``bounds`` (minx, miny, maxx, maxy; metres) at ``px_m`` per pixel.

    Only tiles whose box intersects ``footprint`` are coloured (by score, viridis-like ramp
    over the shown tiles); every other pixel is ``MASK_RGB``, whatever its tile's score.
    """
    from matplotlib import colormaps
    from shapely.geometry import box

    minx, miny, maxx, maxy = bounds
    w, h = max(1, round((maxx - minx) / px_m)), max(1, round((maxy - miny) / px_m))
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = MASK_RGB
    shown = {t: b for t, b in tile_boxes.items() if footprint.intersects(box(*b))}
    if not shown:
        return out
    vals = np.array([tile_scores[t] for t in shown], dtype=float)
    lo, hi = vals.min(), vals.max()
    cmap = colormaps["viridis"]
    for t, (x0, y0, x1, y1) in shown.items():
        u = 0.5 if hi == lo else (tile_scores[t] - lo) / (hi - lo)
        rgb = (np.array(cmap(u)[:3]) * 255).astype(np.uint8)
        c0, c1 = int(np.floor((x0 - minx) / px_m)), int(np.ceil((x1 - minx) / px_m))
        r0, r1 = int(np.floor((maxy - y1) / px_m)), int(np.ceil((maxy - y0) / px_m))
        out[max(r0, 0) : max(min(r1, h), 0), max(c0, 0) : max(min(c1, w), 0)] = rgb
    return out


def _read_chip(path: Path, bounds):
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        bands = (3, 2, 1) if src.count >= 3 else (1,)
        a = src.read(list(bands), window=win, boundless=True, fill_value=np.nan)
    return a.astype("float32")


def write_figures(scores, reference, exclusions, settings: Settings | None = None) -> list[Path]:
    """Write figures (a), (b) and, when a modern stack exists, (c) to outputs/public/figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sbf.report.blind_test import positive_ranks

    s = settings or load_settings()
    out = s.outputs_public / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pr = positive_ranks(scores, reference, exclusions, s)
    paths: list[Path] = []
    if pr:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (aoi, det, layer, size), d in sorted(pr.items()):
            ks, obs, rnd = recall_curve(d["best_ranks"], d["n_tiles"], d["tiles_per_feature"])
            ax.plot(ks, obs, label=f"{det}/{layer}/{aoi}/{size}m", lw=1.2)
            ax.plot(ks, rnd, ls=":", color="grey", lw=0.8)
        ax.set(xlabel="K", ylabel="feature recall@K", title="Recall@K (dotted: random)")
        ax.legend(fontsize=5)
        p = out / "blind_test_recall_curves.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        fig, ax = plt.subplots(figsize=(7, 4))
        for (aoi, det, layer, size), d in sorted(pr.items()):
            ax.hist(
                np.array(d["pos_ranks"]) / d["n_tiles"],
                bins=20,
                range=(0, 1),
                histtype="step",
                label=f"{det}/{layer}/{aoi}/{size}m",
            )
        ax.set(xlabel="relative rank of positive tiles (0 = top)", ylabel="positive tiles")
        ax.legend(fontsize=5)
        p = out / "blind_test_rank_hist.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    paths += _feature_panels(scores, reference, s, out)
    log.info("wrote %d figure(s)", len(paths))
    return paths


def _feature_panels(scores, reference, s: Settings, out: Path) -> list[Path]:
    """Figure (c): per feature, S2 imagery chip and footprint-only score panel (best detector)."""
    import matplotlib.pyplot as plt

    from sbf.detect.harness import reference_footprints
    from sbf.features.tiling import tile_geometries
    from sbf.ingest.modern.process import read_manifest, stack_dir

    paths = []
    for aoi, sub in scores.groupby("aoi_id"):
        man = read_manifest(aoi, s)
        if man is None:
            continue
        s2 = next((n for n in man["layers"] if n.startswith("s2_")), None)
        if s2 is None:
            continue
        crs = s.crs(aoi)
        foot = reference_footprints(aoi, reference, crs)
        det_rows = sub[~sub["detector"].isin(["random", "prior"])]
        for (det, layer, size), g in det_rows.groupby(["detector", "layer", "size_m"]):
            tiles = tile_geometries(list(g["tile_id"]), s.overlap, s).set_index("tile_id")
            sc = g.set_index("tile_id")["score"]
            for _, f in foot.iterrows():
                gm = f.geometry
                if gm.is_empty:
                    continue
                b = gm.buffer(MARGIN_M).bounds
                boxes = {
                    t: tuple(tiles.geometry[t].bounds) for t in tiles.index[tiles.intersects(gm)]
                }
                if not boxes:
                    continue
                px = float(s.modern_resolution_m)
                panel = score_panel(gm, boxes, {t: float(sc[t]) for t in boxes}, b, px)
                fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
                try:
                    chip = _read_chip(stack_dir(aoi, s) / man["layers"][s2]["file"], b)
                    img = np.moveaxis(chip, 0, -1)
                    lo, hi = np.nanpercentile(img, [2, 98])
                    axs[0].imshow(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1).squeeze())
                except Exception as e:  # noqa: BLE001 - imagery is optional
                    log.warning("no imagery chip: %s", type(e).__name__)
                axs[1].imshow(panel)
                axs[0].set_title("imagery", fontsize=8)
                axs[1].set_title(f"{det} score (footprint tiles only)", fontsize=8)
                for a in axs:
                    a.axis("off")
                p = out / f"blind_test_feature_{f['feature_id']}_{det}_{layer}_{size}m.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                paths.append(p)
    return paths
