import csv
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from typer.testing import CliRunner

from sbf.core.crs import lonlat_to_xy
from sbf.core.provenance import read_prov
from sbf.detect.scores import read_scores
from sbf.docs_mining.cli import app
from sbf.docs_mining.gazetteer import load_gazetteer, normalise_alias, resolve_anchor
from sbf.docs_mining.pdf import docs_interim_dir
from sbf.docs_mining.prior import (
    SOURCE_WEIGHTS,
    Claim,
    build_prior,
    build_prior_products,
    claim_surface,
    grid_axes,
    load_accepted_claims,
    parse_bearing,
    resolve_claims,
)
from sbf.docs_mining.review import REVIEW_COLUMNS


def small_grid(center_lonlat=(-0.0572, 26.3117), half_m=6000, res=10.0, crs="EPSG:32631"):
    cx, cy = lonlat_to_xy(*center_lonlat, crs)
    x0 = math.floor((cx - half_m) / res) * res
    y1 = math.ceil((cy + half_m) / res) * res
    n = int(2 * half_m / res)
    return Affine(res, 0, x0, 0, -res, y1), n, n, crs


def claim(cid="c1", anchor="Gerboise Verte", d=1.9, dsig=0.3, b="SE", bsig=10.0, src="official",
          ref=False):  # fmt: skip
    return Claim(cid, anchor, d, dsig, b, bsig, src, ref)


def write_review(settings, rows):
    d = docs_interim_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "review.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            base = dict.fromkeys(REVIEW_COLUMNS, "")
            base.update(quote="synthetic", object_buried="unknown", doc_id="d", page=1)
            base.update({k: ("" if v is None else v) for k, v in r.items()})
            w.writerow(base)
    return p


# ------------------------------------------------------------------ compass
@pytest.mark.parametrize(
    ("text", "deg"),
    [
        ("N", 0), ("NNE", 22.5), ("NE", 45), ("ENE", 67.5), ("E", 90), ("ESE", 112.5),
        ("SE", 135), ("SSE", 157.5), ("S", 180), ("SSW", 202.5), ("SW", 225), ("WSW", 247.5),
        ("W", 270), ("WNW", 292.5), ("NW", 315), ("NNW", 337.5),
        ("north-east", 45), ("North East", 45), ("northeast", 45), ("south-south-west", 202.5),
        ("east north east", 67.5), ("to the north-west", 315),
        ("nord", 0), ("nord-est", 45), ("Nord-Nord-Est", 22.5), ("est-nord-est", 67.5),
        ("sud-ouest", 225), ("ouest", 270), ("O", 270), ("ONO", 292.5), ("SSO", 202.5),
        ("au nord-ouest", 315), ("vers l'est", 90), ("sud-est", 135), ("120", 120),
        ("45°", 45), ("365", 5),
    ],
)  # fmt: skip
def test_parse_bearing(text, deg):
    assert parse_bearing(text) == pytest.approx(deg)


@pytest.mark.parametrize("text", [None, "", "nearby", "NEN", "somewhere", "ENEE"])
def test_parse_bearing_rejects(text):
    assert parse_bearing(text) is None


# ------------------------------------------------------------------ gazetteer
def test_alias_normalisation():
    assert normalise_alias("GZ-1") == normalise_alias("gz 1") == normalise_alias("GZ1")
    assert normalise_alias("Point Zéro") == normalise_alias("point zero")


@pytest.mark.parametrize(
    ("text", "aid"),
    [
        ("Gerboise Bleue", "GZ1"), ("GZ Gerboise bleue", "GZ1"), ("point zéro", "GZ1"),
        ("gerboise-bleue", "GZ1"), ("GZ-4", "GZ4"), ("Gerboise Verte", "GZ4"),
        ("ground zero of Gerboise Blanche", "GZ2"), ("Augias", "GZ3"),
        ("Reggane", "REGGANE"), ("oasis de Reggane", "REGGANE"), ("Béryl", "TA1"),
    ],
)  # fmt: skip
def test_resolve_anchor(tmp_project, text, aid):
    a = resolve_anchor(text, tmp_project)
    assert a is not None and a.anchor_id == aid and a.resolved


def test_unresolved_and_unknown(tmp_project):
    assert resolve_anchor("Mount Nowhere", tmp_project) is None
    base = resolve_anchor("Hamoudia", tmp_project)
    assert base is not None and not base.resolved and base.needs_human_location
    rc = resolve_claims(
        [claim("a", "Hamoudia"), claim("b", "Mount Nowhere"), claim("c", "GZ1"),
         claim("d", "Tan Afella")], "REG", settings=tmp_project,
    )  # fmt: skip
    assert [r.status for r in rc] == [
        "anchor_needs_location",
        "unknown_anchor",
        "ok",
        "other_aoi",
    ]


def test_gazetteer_has_only_public_points(tmp_project):
    gz = load_gazetteer(tmp_project)
    assert {a.anchor_id for a in gz.anchors} == {"GZ1", "GZ2", "GZ3", "GZ4", "TA1", "PO1",
                                                  "BASE", "REGGANE"}  # fmt: skip
    assert not gz.get("BASE").resolved


# ------------------------------------------------------------------ surfaces
def _peak_polar(grid, surf, anchor_xy):
    xs, ys = grid_axes(grid)
    rs, cs, w = surf
    i, j = np.unravel_index(np.argmax(w), w.shape)
    dx, dy = xs[cs][j] - anchor_xy[0], ys[rs][i] - anchor_xy[1]
    return math.hypot(dx, dy) / 1000, math.degrees(math.atan2(dx, dy)) % 360


def test_sector_shape_and_unit_mass():
    grid = small_grid(half_m=9000, res=20.0)
    xs, ys = grid_axes(grid)
    axy = (float(xs[len(xs) // 2]), float(ys[len(ys) // 2]))
    rs, cs, w = claim_surface(xs, ys, axy, distance_km=5.0, distance_sigma_km=0.5,
                              azimuth_deg=45.0, bearing_sigma_deg=15.0)  # fmt: skip
    assert w.sum() == pytest.approx(1.0)
    r, brg = _peak_polar(grid, (rs, cs, w), axy)
    assert r == pytest.approx(5.0, abs=0.1) and brg == pytest.approx(45.0, abs=3.0)
    # mass concentrated in the sector: opposite bearing has ~nothing
    X, Y = np.meshgrid(xs[cs] - axy[0], ys[rs] - axy[1])
    ang = np.degrees(np.arctan2(X, Y)) % 360
    inside = np.abs((ang - 45 + 180) % 360 - 180) < 45
    assert w[inside].sum() > 0.95


def test_annulus_and_isotropic():
    grid = small_grid(half_m=9000, res=20.0)
    xs, ys = grid_axes(grid)
    axy = (float(xs[len(xs) // 2]), float(ys[len(ys) // 2]))
    rs, cs, w = claim_surface(xs, ys, axy, distance_km=4.0, distance_sigma_km=0.4)
    assert w.sum() == pytest.approx(1.0)
    X, Y = np.meshgrid(xs[cs] - axy[0], ys[rs] - axy[1])
    ang = np.degrees(np.arctan2(X, Y)) % 360
    quadrant = [w[(ang >= q) & (ang < q + 90)].sum() for q in (0, 90, 180, 270)]
    assert max(quadrant) - min(quadrant) < 0.05  # rotationally symmetric
    r_peak, _ = _peak_polar(grid, (rs, cs, w), axy)
    assert r_peak == pytest.approx(4.0, abs=0.15)
    # default sigma = 50 % of distance
    _, _, w2 = claim_surface(xs, ys, axy, distance_km=4.0)
    assert w2.max() < w.max()
    # neither: isotropic gaussian, sigma 2 km, peaks at the anchor; bearing alone is ignored
    rs, cs, w3 = claim_surface(xs, ys, axy, distance_km=None)
    _, _, w4 = claim_surface(xs, ys, axy, distance_km=None, azimuth_deg=90.0)
    assert w3.sum() == pytest.approx(1.0) and np.array_equal(w3, w4)
    i, j = np.unravel_index(np.argmax(w3), w3.shape)
    assert math.hypot(xs[cs][j] - axy[0], ys[rs][i] - axy[1]) < 25
    X, Y = np.meshgrid(xs[cs] - axy[0], ys[rs] - axy[1])
    r2 = np.hypot(X, Y)
    sigma = math.sqrt((w3 * r2**2).sum() / 2)
    assert sigma == pytest.approx(2000, rel=0.05)


def test_off_grid_claim_is_none():
    grid = small_grid(half_m=1000, res=20.0)
    xs, ys = grid_axes(grid)
    assert claim_surface(xs, ys, (xs[0] + 1e6, ys[0]), distance_km=1.0) is None


# ------------------------------------------------------------------ weighting, floor
def test_weighting_floor_and_range(tmp_project):
    grid = small_grid(half_m=5000, res=20.0)
    # two identical-shape claims at opposite sides (anchors GZ1 and GZ4 about 2 km apart)
    rcs = resolve_claims(
        [claim("off", "GZ1", d=None, dsig=None, b=None, src="official"),
         claim("sec", "GZ4", d=None, dsig=None, b=None, src="secondary")],
        "REG", settings=tmp_project,
    )  # fmt: skip
    prior, used, off = build_prior(rcs, grid, eps=0.02)
    assert used == ["off", "sec"] and off == []
    assert prior.dtype == np.float32
    assert prior.max() == pytest.approx(1.0) and prior.min() >= 0.02 - 1e-7
    assert prior.min() > 0.0  # uniform floor, never zero
    xs, ys = grid_axes(grid)
    crs = grid[3]

    def at(lonlat):
        x, y = lonlat_to_xy(*lonlat, crs)
        return prior[int(np.argmin(abs(ys - y))), int(np.argmin(abs(xs - x)))]

    gz1, gz4 = at((-0.0572, 26.3117)), at((-0.0733, 26.3217))
    # equal unit mass x weight 1.0 vs 0.4 (the two gaussians overlap slightly)
    assert gz1 > gz4 > 0.02
    assert SOURCE_WEIGHTS == {"official": 1.0, "testimony": 0.7, "secondary": 0.4}


def test_unresolved_only_raises(tmp_project):
    rcs = resolve_claims([claim("x", "Nowhere")], "REG", settings=tmp_project)
    with pytest.raises(ValueError):
        build_prior(rcs, small_grid(half_m=1000, res=20.0))


# ------------------------------------------------------------------ products + CLI
def _row(cid, anchor, accept="y", **kw):
    base = {"extraction_id": cid, "anchor": anchor, "distance_km": 2, "distance_uncertainty_km": 0.5,
            "bearing": "SE", "bearing_uncertainty_deg": 20, "confidence_source": "official",
            "describes_reference_feature": "False", "accept": accept}  # fmt: skip
    base.update(kw)
    return base


def test_load_accepted_only(tmp_project):
    write_review(tmp_project, [_row("a", "GZ1"), _row("b", "GZ1", accept="n"),
                               _row("c", "GZ1", accept="")])  # fmt: skip
    cl = load_accepted_claims(tmp_project)
    assert [c.extraction_id for c in cl] == ["a"] and cl[0].distance_km == 2.0


def test_prior_products(tmp_project):
    cl = [Claim("a", "GZ4", 1.9, 0.3, "SE", 10, "official", False)]
    r = build_prior_products("REG", tmp_project, claims=cl, grid=small_grid(half_m=4000, res=10.0))
    with rasterio.open(r.raster) as src:
        a = src.read(1)
        assert src.crs.to_string() == "EPSG:32631" and a.max() == pytest.approx(1.0)
    prov = read_prov(r.raster)
    assert any("extraction_ids=a" in s for s in prov["processing_steps"])
    t = read_scores("REG", "prior", "docs_prior", tmp_project)
    assert set(t["size_m"]) == {64, 128} and len(t) > 0
    assert t["score_norm"].between(0, 1).all() and "geometry" not in t.columns


def test_prior_cli_refuses_without_accepted(tmp_project):
    write_review(tmp_project, [_row("a", "GZ1", accept="")])
    res = CliRunner().invoke(app, ["prior", "--aoi", "REG"])
    assert res.exit_code == 1 and "human gate" in res.output
    assert not (tmp_project.data_processed / "prior").exists()


def test_anchors_cli(tmp_project):
    write_review(tmp_project, [_row("a", "Gerboise Bleue"), _row("b", "Mount Nowhere"),
                               _row("c", "Hamoudia"), _row("d", "GZ1", accept="n")])  # fmt: skip
    res = CliRunner().invoke(app, ["anchors"])
    assert res.exit_code == 0
    assert "a  resolved -> GZ1" in res.output
    assert "UNRESOLVED (unknown anchor)" in res.output
    assert "needs_human_location" in res.output
    assert "26." not in res.output  # no coordinates
