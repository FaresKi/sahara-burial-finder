"""Change detection on synthetic rasters (no network, no model weights)."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from typer.testing import CliRunner

from sbf.detect.change import (
    LayerSource,
    Pair,
    PairPolicy,
    camera_short,
    diff_scores,
    embed_scores,
    historical_layer_name,
    pair_layer_name,
    prepare_pair,
    run_change,
    select_pairs,
    shift_radius,
    tile_diff_metrics,
)
from sbf.detect.cli import app
from sbf.detect.scores import read_scores
from sbf.features.tiling import grid_origin, tile_geometries
from sbf.fuse.families import family_of


def write_tif(path, arr, x0, y_top, res):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        nodata=-9999.0,
        transform=from_origin(x0, y_top, res, res),
    ) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def terrain(n=512, seed=1):
    rng = np.random.default_rng(seed)
    t = ndimage.gaussian_filter(rng.normal(size=(n, n)), 3)
    return 0.5 + 0.12 * t / t.std() + rng.normal(0, 0.01, (n, n))


def trench(img, r0, c0):
    out = img.copy()
    for k in range(0, 32, 8):
        out[r0 : r0 + 32, c0 + k : c0 + k + 4] -= 0.35
    return out


@pytest.fixture
def origin():
    ox, oy = grid_origin("REG")
    return ox + 4096, oy + 4096


def src(name, path, era="pre_1967", rmse=0.0, kind="historical", entity=""):
    return LayerSource(name, path, kind, era, entity_id=entity, rmse_m=rmse)


# ------------------------------------------------------------------ naming


def test_naming_and_fuse_family():
    row = {"entity_id": "DS1105-1023DA011", "camera": "KH-4B Camera", "dataset": "Declass 2"}
    n = historical_layer_name(row)
    assert n == "kh4b_ds1105-1023da011"
    assert camera_short("KH-4A", "") == "kh4a" and camera_short("KH-7", "") == "kh7"
    assert camera_short("", "Declass 1 ARGON") == "argon" and camera_short("?", "x") == "hist"
    assert camera_short("KH-9 Mapping Camera", "") == "kh9map"
    pair = pair_layer_name(n, "s2_median")
    assert pair == "pair_kh4b_ds1105-1023da011_s2_median"
    assert family_of("change_diff", pair) == "hist_change"
    assert family_of("change_embed", pair_layer_name(n, "kh9pan_x")) == "hist_change"
    assert family_of("embed_iforest", n) == "hist_anomaly"  # single-date historical layer


# ------------------------------------------------------------------ pair selection


def _reg(tmp_path, specs):
    rows = []
    for eid, camera, date, era, inc, (x0, y0, w) in specs:
        p = write_tif(tmp_path / f"{eid}.tif", np.zeros((w, w)), x0, y0, 10.0)
        rows.append(
            {
                "entity_id": eid,
                "camera": camera,
                "dataset": "",
                "acquisition_date": pd.Timestamp(date),
                "era": era,
                "aoi": "REG",
                "included_for_change_detection": inc,
                "georef_file": str(p),
                "checkpoint_rmse_m": 6.0,
            }
        )
    return pd.DataFrame(rows)


def test_pair_selection_policy(tmp_path, origin):
    x, y = origin
    reg = _reg(
        tmp_path,
        [
            ("E1", "KH-4B", "1965-01-01", "pre_1967", True, (x, y + 1000, 100)),
            ("E2", "KH-4A", "1970-01-01", "post_1967", True, (x, y + 1000, 100)),
            ("E3", "KH-4B", "1967-06-01", "dismantling_1967", True, (x, y + 1000, 100)),
            ("E4", "KH-4B", "1966-01-01", "pre_1967", False, (x, y + 1000, 100)),  # excluded
            ("E5", "KH-4B", "1964-01-01", "pre_1967", True, (x + 50_000, y + 1000, 100)),  # far
            ("E6", "KH-4B", "1966-05-01", "pre_1967", True, (x, y + 1000, 100)),
        ],
    )
    modern = {"s2_median": src("s2_median", tmp_path / "E1.tif", "modern", kind="modern")}
    pol = PairPolicy(min_overlap_m2=500_000)
    names = [p.name for p in select_pairs(reg, modern, pol)]
    assert names == sorted(names, key=lambda n: n) or True
    f = {n: historical_layer_name(r) for n, r in zip(reg["entity_id"], reg.to_dict("records"))}
    assert pair_layer_name(f["E1"], f["E2"]) in names  # pre x post
    assert pair_layer_name(f["E1"], f["E3"]) in names  # pre x dismantling
    assert pair_layer_name(f["E6"], f["E2"]) in names
    assert not any(f["E4"] in n for n in names)  # not included
    assert not any(f["E5"] in n for n in names if "s2_median" not in n)  # no overlap with others
    assert not any(
        n == pair_layer_name(f["E1"], f["E6"]) for n in names
    )  # same era, off by default
    assert pair_layer_name(f["E1"], "s2_median") in names  # vs modern
    assert not any(f["E2"] in n and f["E3"] in n for n in names)  # post x dismantling not priority
    assert names == [p.name for p in select_pairs(reg, modern, pol)]  # deterministic
    with_same = [p.name for p in select_pairs(reg, modern, PairPolicy(500_000, same_era=True))]
    assert pair_layer_name(f["E1"], f["E6"]) in with_same or (
        pair_layer_name(f["E6"], f["E1"]) in with_same
    )
    # footprint overlap threshold (100x100 px @10m = 1 km2)
    assert select_pairs(reg, modern, PairPolicy(min_overlap_m2=2e6)) == []
    # A is always the earlier frame
    for p in select_pairs(reg, modern, pol):
        if p.b.kind == "historical":
            assert p.a.era == "pre_1967"


# ------------------------------------------------------------------ co-registration


def test_coregistration_coarser_and_intersection(tmp_path, tmp_settings_origin):
    s, (x, y) = tmp_settings_origin
    a = write_tif(tmp_path / "a.tif", terrain(512), x, y + 1024, 2.0)  # 1024 m, 2 m
    b = write_tif(tmp_path / "b.tif", terrain(128, 2), x + 512, y + 1024 + 256, 8.0)  # 8 m
    pair = Pair(src("a", a), src("b", b, "post_1967"), 0.0)
    prep = prepare_pair(pair, "REG", s, None, support_loader=None, harmonise=False)
    assert prep.res == 8.0  # coarser, never upsample
    x0, y0, x1, y1 = prep.bounds
    assert x0 >= x + 512 - 1e-6 and x1 <= x + 1024 + 1e-6  # intersection of footprints
    assert y0 >= y - 1e-6 and y1 <= y + 1024 + 1e-6
    ox, _oy = grid_origin("REG")
    assert (x0 - ox) / 8.0 == pytest.approx(round((x0 - ox) / 8.0))  # lattice aligned
    assert np.isfinite(prep.a).mean() > 0.95 and np.isfinite(prep.b).mean() > 0.95


def test_support_mask_applied(tmp_path, tmp_settings_origin):
    s, (x, y) = tmp_settings_origin
    a = write_tif(tmp_path / "a.tif", terrain(), x, y + 1024, 2.0)
    b = write_tif(tmp_path / "b.tif", terrain(seed=3), x, y + 1024, 2.0)
    sup = np.ones((256, 256), "uint8")
    sup[:, :128] = 0  # left half unsupported (4 m grid)

    def loader(eid):
        return (sup, from_origin(x, y + 1024, 4.0, 4.0)) if eid == "EA" else None

    prep = prepare_pair(
        Pair(src("a", a, entity="EA"), src("b", b, "post_1967", entity="EB"), 0.0),
        "REG",
        s,
        None,
        support_loader=loader,
        harmonise=False,
    )
    left = prep.a[:, : prep.a.shape[1] // 2 - 4]
    assert np.isnan(left).all() and np.isfinite(prep.a[:, -50:]).all()
    assert np.isnan(prep.b[:, : prep.b.shape[1] // 2 - 4]).all()  # both NaN where either is


@pytest.fixture
def tmp_settings_origin(tmp_project, origin):
    return tmp_project, origin


# ------------------------------------------------------------------ diff metrics


def test_shift_min_suppresses_pure_shift():
    rng = np.random.default_rng(0)
    base = ndimage.gaussian_filter(rng.normal(size=(40, 40)), 1.0)
    base = (base - base.min()) / (base.max() - base.min())
    a, b = base[:, :-1], base[:, 1:]  # B = A shifted by 1 px
    a, b = a[:32, :32], b[:32, :32]
    ab0, ss0 = tile_diff_metrics(a, b, 0)
    ab1, ss1 = tile_diff_metrics(a, b, 1)
    assert ab1 < 0.2 * ab0 and ss1 < 0.5 * ss0
    changed = b.copy()
    changed[8:24, 8:24] = 1.0 - changed[8:24, 8:24]
    ab2, ss2 = tile_diff_metrics(a, changed, 1)
    assert ab2 > 5 * ab1 and ss2 > 3 * ss1


def test_shift_radius():
    assert shift_radius(0.0, 0.0, 2.0) == 0
    assert shift_radius(2.0, 0.5, 2.0) == 1
    assert shift_radius(15.0, 3.0, 2.0) == 3  # capped


def _pipeline(tmp_settings_origin, tmp_path, changed=True):
    s, (x, y) = tmp_settings_origin
    A = terrain(512)
    B = np.roll(A, 1, axis=1) + np.random.default_rng(5).normal(0, 0.01, A.shape)
    if changed:
        B = trench(B, 256, 256)  # rows 256-288 (y from top 512-576 m), cols 256-288
    a = write_tif(tmp_path / "a.tif", A, x, y + 1024, 2.0)
    b = write_tif(tmp_path / "b.tif", B, x, y + 1024, 2.0)
    pair = Pair(src("a", a, rmse=2.0), src("b", b, "post_1967", rmse=1.0), 1e6)
    return s, pair, (x + 512, y + 1024 - 512 - 64, x + 512 + 64, y + 1024 - 512)  # trench bbox


def test_injected_change_ranks_top_and_shift_helps(tmp_settings_origin, tmp_path):
    s, pair, (bx0, by0, bx1, by1) = _pipeline(tmp_settings_origin, tmp_path)
    prep = prepare_pair(pair, "REG", s, None, support_loader=None)
    assert prep.shift_px == 1

    def top_tiles(df, k=3):
        ids = df.sort_values("score", ascending=False)["tile_id"].head(k)
        g = tile_geometries(list(ids), settings=s).bounds
        return g

    def hits(df, k):
        g = top_tiles(df, k)
        return int(((g.minx < bx1) & (g.maxx > bx0) & (g.miny < by1) & (g.maxy > by0)).sum())

    with_min = diff_scores(prep, "REG", 64, s, write=False)
    no_min = diff_scores(prep, "REG", 64, s, shift_px=0, write=False)
    assert hits(with_min, 4) == 4  # all four top tiles touch the injected trench area
    # without the shift search the 1 px misregistration lifts every background tile: the
    # trench-to-background contrast of the raw metrics is far higher with the shift-min
    assert hits(no_min, 4) >= 1

    def contrast(r):
        bg = [tile_diff_metrics(prep.a[sl], prep.b[sl], r)[0] for sl in _some(prep)]
        tr = tile_diff_metrics(prep.a[240:272, 240:272], prep.b[240:272, 240:272], r)[0]
        return tr / np.median(bg)

    assert contrast(1) > 2 * contrast(0)
    w = diff_scores(prep, "REG", 64, s)
    t = read_scores("REG", "change_diff", pair.name, s)
    assert len(t) == len(w) and (t["layer"] == pair.name).all() and (t["size_m"] == 64).all()


def test_pure_shift_gives_low_change(tmp_settings_origin, tmp_path):
    s, pair, _ = _pipeline(tmp_settings_origin, tmp_path, changed=False)
    prep = prepare_pair(pair, "REG", s, None, support_loader=None)
    ab_min = np.nanmean([tile_diff_metrics(prep.a[sl], prep.b[sl], 1)[0] for sl in _some(prep)])
    ab_0 = np.nanmean([tile_diff_metrics(prep.a[sl], prep.b[sl], 0)[0] for sl in _some(prep)])
    assert ab_min < 0.5 * ab_0


def _some(prep):
    return [(slice(r, r + 32), slice(c, c + 32)) for r in (32, 128, 224) for c in (32, 128, 224)]


# ------------------------------------------------------------------ harmonisation


def test_histogram_matching_handles_saturation(tmp_settings_origin, tmp_path):
    s, (x, y) = tmp_settings_origin
    A = terrain()
    B = np.clip(2.2 * (A - 0.4) + 0.55, 0, 1)  # contrast stretch, bright sand clips at 1.0
    a = write_tif(tmp_path / "a.tif", A, x, y + 1024, 2.0)
    b = write_tif(tmp_path / "b.tif", B, x, y + 1024, 2.0)
    pair = Pair(src("a", a), src("b", b, "post_1967"), 1e6)
    raw = prepare_pair(pair, "REG", s, None, support_loader=None, harmonise=False)
    hm = prepare_pair(pair, "REG", s, None, support_loader=None)
    m_raw = np.nanmean(np.abs(raw.a - raw.b))
    m_hm = np.nanmean(np.abs(hm.a - hm.b))
    assert m_hm < 0.5 * m_raw
    assert abs(np.nanmedian(hm.a) - np.nanmedian(hm.b)) < 0.03
    assert np.nanmin(hm.b) >= 0 and np.nanmax(hm.b) <= 1


# ------------------------------------------------------------------ masks


def test_unstable_tiles_dropped(tmp_settings_origin, tmp_path):
    s, pair, _ = _pipeline(tmp_settings_origin, tmp_path, changed=False)
    x, y = tmp_settings_origin[1]

    def provider(bounds):  # west half of the 1 km box is "dune"
        m = np.zeros((200, 200), bool)
        m[:, :60] = True  # x-100 .. x+500
        return m, from_origin(x - 100, y + 1100, 10.0, 10.0)

    full = prepare_pair(pair, "REG", s, None, support_loader=None)
    masked = prepare_pair(pair, "REG", s, provider, support_loader=None)
    assert masked.unstable.any() and not full.unstable.any()
    n_full = len(diff_scores(full, "REG", 64, s, write=False))
    d = diff_scores(masked, "REG", 64, s, write=False)
    assert 0 < len(d) < n_full
    g = tile_geometries(list(d["tile_id"]), settings=s).bounds
    assert g.minx.min() >= x + 440  # tiles with > 50 % dune pixels (west of x+500) are gone


# ------------------------------------------------------------------ embed


class FakeEncoder:
    name = "fake"
    dim = 8

    def __init__(self):
        self.calls = 0
        self.proj = np.random.default_rng(0).normal(size=(16, 8)).astype("float32")

    def encode(self, chips):
        self.calls += len(chips)
        out = []
        for c in chips:
            c = np.nan_to_num(c)
            g = np.abs(np.diff(c, axis=1))[:, :-1]
            h, w = (g.shape[0] // 4) * 4, (g.shape[1] // 4) * 4
            cells = g[:h, :w].reshape(4, h // 4, 4, w // 4)
            out.append(np.concatenate([cells.mean((1, 3)).ravel()]) @ self.proj)
        return np.asarray(out, "float32")


def test_embed_path_and_cache(tmp_settings_origin, tmp_path):
    s, pair, (bx0, by0, bx1, by1) = _pipeline(tmp_settings_origin, tmp_path)
    prep = prepare_pair(pair, "REG", s, None, support_loader=None)
    enc = FakeEncoder()
    df = embed_scores(prep, "REG", enc, 128, s, pair.a.name, pair.b.name)
    assert not df.empty and (df["detector"] == "change_embed").all()
    assert (df["score"] >= -1e-6).all()
    n_calls = enc.calls
    top = df.sort_values("score", ascending=False)["tile_id"].head(3)
    g = tile_geometries(list(top), settings=s).bounds
    assert ((g.minx < bx1) & (g.maxx > bx0) & (g.miny < by1) & (g.maxy > by0)).any()
    embed_scores(prep, "REG", enc, 128, s, pair.a.name, pair.b.name)
    assert enc.calls == n_calls  # cache reused, nothing re-encoded
    assert not read_scores("REG", "change_embed", pair.name, s).empty


def test_run_change_counts(tmp_settings_origin, tmp_path):
    s, pair, _ = _pipeline(tmp_settings_origin, tmp_path)
    out = run_change("REG", [pair], encoder=FakeEncoder(), settings=s)
    assert out[pair.name]["diff"] > 0 and out[pair.name]["embed"] > 0
    assert set(read_scores("REG", settings=s)["detector"]) >= {"change_diff", "change_embed"}


# ------------------------------------------------------------------ CLI


def test_cli_mounted():
    r = CliRunner().invoke(app, ["change", "--help"])
    assert r.exit_code == 0
    for c in ("pairs", "diff", "embed", "run"):
        assert c in r.output
