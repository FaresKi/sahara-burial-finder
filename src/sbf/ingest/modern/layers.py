"""Layer and raw-product registry for the modern baseline stack.

Layer names are the `layer` vocabulary of the ARCHITECTURE score table.
Raw products are what the exporter writes to `data/raw/modern/{AOI}/{product}.tif`;
layers are bands (or derivations) of those products in `data/processed/modern/{AOI}/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NODATA_RAW = -9999.0

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S2_MEANING = {
    "B2": "blue 490 nm",
    "B3": "green 560 nm",
    "B4": "red 665 nm",
    "B8": "NIR 842 nm",
    "B11": "SWIR1 1610 nm",
    "B12": "SWIR2 2190 nm",
}
S2_SEASONS = {"djf": (12, 1, 2), "mam": (3, 4, 5), "jja": (6, 7, 8), "son": (9, 10, 11)}


@dataclass(frozen=True)
class ProductSpec:
    """A raw export product: one multi-band GeoTIFF."""

    key: str
    bands: tuple[str, ...]  # output band names, in order
    dataset_candidates: tuple[str, ...]  # tried in order, first one verified at runtime wins
    description: str
    native_resolution_m: float
    units: str


PRODUCTS: dict[str, ProductSpec] = {}
for _key in ("median", *S2_SEASONS):
    _k = "s2_" + _key
    PRODUCTS[_k] = ProductSpec(
        _k,
        tuple(S2_BANDS),
        ("COPERNICUS/S2_SR_HARMONIZED",),
        "Sentinel-2 L2A harmonized, SCL/QA60 cloud-masked median"
        + ("" if _key == "median" else f" ({_key.upper()} months only)"),
        10.0,
        "surface reflectance (0-1)",
    )
PRODUCTS["s1"] = ProductSpec(
    "s1",
    ("VV_mean", "VV_std", "VH_mean", "VH_std"),
    ("COPERNICUS/S1_GRD",),
    "Sentinel-1 GRD IW, per-pixel temporal mean and std of dB backscatter",
    10.0,
    "dB",
)
PRODUCTS["palsar_latest"] = ProductSpec(
    "palsar_latest",
    ("HH", "HV"),
    ("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH", "JAXA/ALOS/PALSAR/YEARLY/SAR"),
    "ALOS PALSAR yearly mosaic, most recent year, raw DN",
    25.0,
    "DN (converted to dB gamma0 in build-stack)",
)
PRODUCTS["palsar_multi"] = ProductSpec(
    "palsar_multi",
    ("HH", "HV"),
    ("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH", "JAXA/ALOS/PALSAR/YEARLY/SAR"),
    "ALOS PALSAR yearly mosaics, multi-year mean DN",
    25.0,
    "DN (converted to dB gamma0 in build-stack)",
)
PRODUCTS["dem"] = ProductSpec(
    "dem",
    ("DEM",),
    ("COPERNICUS/DEM/GLO30_2024_1", "COPERNICUS/DEM/GLO30"),  # 2024_1 supersedes GLO30
    "Copernicus DEM GLO-30 mosaic",
    30.0,
    "m",
)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    product: str  # raw product key
    bands: tuple[int, ...]  # 1-based band indices in the raw product
    band_names: tuple[str, ...]
    band_meaning: tuple[str, ...]
    units: str
    resampling: str  # "bilinear" | "nearest"
    kind: str = "raw"  # raw | palsar_db | slope | roughness
    derived_from: str | None = None
    extra: dict = field(default_factory=dict)


def _build_layers() -> dict[str, LayerSpec]:
    out: dict[str, LayerSpec] = {}
    for key in ("median", *S2_SEASONS):
        n = "s2_" + key
        out[n] = LayerSpec(
            n,
            n,
            tuple(range(1, 7)),
            tuple(S2_BANDS),
            tuple(S2_MEANING[b] for b in S2_BANDS),
            "surface reflectance (0-1)",
            "bilinear",
        )
    for pol in ("vv", "vh"):
        for stat in ("mean", "std"):
            idx = PRODUCTS["s1"].bands.index(f"{pol.upper()}_{stat}") + 1
            n = f"s1_{pol}_{stat}"
            out[n] = LayerSpec(
                n,
                "s1",
                (idx,),
                (f"{pol.upper()}_{stat}",),
                (f"{pol.upper()} temporal {stat} of dB backscatter",),
                "dB",
                "bilinear",
            )
    for pol, idx in (("hh", 1), ("hv", 2)):
        for suffix, prod in (("", "palsar_latest"), ("_multi", "palsar_multi")):
            n = f"palsar_{pol}{suffix}"
            what = "latest year" if not suffix else "multi-year mean (of DN)"
            out[n] = LayerSpec(
                n,
                prod,
                (idx,),
                (pol.upper(),),
                (f"L-band {pol.upper()} gamma0, {what}",),
                "dB",
                "bilinear",
                kind="palsar_db",
            )
    out["dem"] = LayerSpec(
        "dem", "dem", (1,), ("DEM",), ("Copernicus GLO-30 elevation",), "m", "bilinear"
    )
    out["dem_slope"] = LayerSpec(
        "dem_slope",
        "dem",
        (1,),
        ("slope",),
        ("slope from DEM, central differences",),
        "degrees",
        "bilinear",
        kind="slope",
        derived_from="dem",
    )
    out["dem_roughness"] = LayerSpec(
        "dem_roughness",
        "dem",
        (1,),
        ("roughness",),
        ("std of elevation in a 5x5 pixel window",),
        "m",
        "bilinear",
        kind="roughness",
        derived_from="dem",
    )
    return out


LAYERS: dict[str, LayerSpec] = _build_layers()
