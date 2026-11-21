"""Tile encoders for the embedding anomaly detector (SPEC Phase 4).

``TileEncoder`` is the whole interface the detector needs: ``encode(chips) -> (n, d)`` float32,
plus ``name`` (used in cache paths) and ``dim``. `DinoV2Encoder` is the real model; tests inject a
tiny fake through the same Protocol. torch / transformers are imported lazily, only when the real
encoder is first used. Model weights are downloaded from the Hugging Face hub on first use (human
only; never in tests).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from sbf.features.preprocess import METHODS as PREPROCESS_METHODS
from sbf.features.preprocess import preprocess_array, robust_range, to_unit

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")
INPUT_SIZE = 224  # multiple of the DINOv2 patch size (14)
MODEL_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
}


@runtime_checkable
class TileEncoder(Protocol):
    """Chips (float32, NaN = nodata; (n, n) single band or (bands, n, n)) -> (n_chips, dim)."""

    name: str
    dim: int

    def encode(self, chips: list[np.ndarray]) -> np.ndarray: ...


def chip_to_rgb(chip: np.ndarray, preprocess: str = "clahe", size: int = INPUT_SIZE) -> np.ndarray:
    """Normalise one chip to a ``(3, size, size)`` ImageNet-normalised float32 array.

    Single-band chips go through `sbf.features.preprocess` (``clahe`` default, suited to saturated
    historical scans; ``lcn``; ``none`` = robust 2-98 % stretch), NaN is filled with the chip median,
    and the band is replicated to 3 channels. 3-band chips are robust-stretched per band. The result
    is bilinearly resized to ``size`` x ``size`` (``size`` must be a multiple of 14).
    """
    from skimage.transform import resize

    if preprocess not in PREPROCESS_METHODS:
        raise ValueError(f"unknown preprocess {preprocess!r}; choose from {PREPROCESS_METHODS}")
    a = np.asarray(chip, dtype="float32")
    if a.ndim == 2:
        a = a[None]
    bands = []
    for b in a:
        if preprocess == "none":
            p = to_unit(b, *robust_range(b))
        else:
            p = preprocess_array(b, preprocess)
            if preprocess == "lcn":
                p = np.clip(p / 6.0 + 0.5, 0.0, 1.0)
        fill = np.nanmedian(p) if np.isfinite(p).any() else 0.5
        bands.append(np.nan_to_num(p, nan=float(fill)))
    if len(bands) == 1:
        bands = bands * 3
    elif len(bands) != 3:
        raise ValueError("chips must have 1 or 3 bands")
    img = np.stack(bands, axis=-1)
    img = resize(img, (size, size), order=1, anti_aliasing=img.shape[0] > size).astype("float32")
    return ((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def pick_device(device: str | None = None) -> str:
    """``mps`` if available, else ``cpu`` (an explicit ``device`` wins). Lazy torch import."""
    if device:
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class DinoV2Encoder:
    """DINOv2 via transformers ``AutoModel``; embedding = [CLS token ; mean of patch tokens].

    The two vectors are concatenated, so ``dim = 2 * hidden_size`` (768 for dinov2-small):
    CLS carries global semantics, the patch mean keeps texture/local-structure information that
    matters for subtle ground disturbance. Inference is float32 (fp16 on MPS is not reliably
    supported by all ops, so it is off), batched, under ``torch.inference_mode``.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        device: str | None = None,
        preprocess: str = "clahe",
        batch_size: int = 32,
    ) -> None:
        self.model_id = MODEL_IDS.get(model_id, model_id)
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.device = pick_device(device)
        short = self.model_id.rsplit("/", 1)[-1]
        self.name = f"{short}_{preprocess}"
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = 2 * int(self._load().config.hidden_size)
        return self._dim

    def _load(self):
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        return self._model

    def encode(self, chips: list[np.ndarray]) -> np.ndarray:
        import torch

        model = self._load()
        out = []
        for i in range(0, len(chips), self.batch_size):
            batch = np.stack(
                [chip_to_rgb(c, self.preprocess) for c in chips[i : i + self.batch_size]]
            )
            with torch.inference_mode():
                h = model(pixel_values=torch.from_numpy(batch).to(self.device)).last_hidden_state
                feat = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
            out.append(feat.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype="float32")
        return np.concatenate(out).astype("float32")


def get_encoder(
    name: str = "dinov2-small", device: str | None = None, preprocess: str = "clahe"
) -> TileEncoder:
    """Build an encoder from a short name (``dinov2-small``, ``dinov2-base``) or a HF model id."""
    return DinoV2Encoder(name, device=device, preprocess=preprocess)
