"""Per-frame fidelity metrics and the motion-aware warping error.

The central design choice: motion is estimated on the REFERENCE video and
then applied to the reconstruction. Flicker violates brightness constancy,
the assumption every optical-flow method rests on; estimating flow on the
flickering output would measure the defect with an instrument the defect
corrupts. The reference gives a clean motion field, and the question becomes:
"does the reconstruction move the way the scene moves?"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity

from .flow import estimate_flow
from .warp import backward_warp, fb_consistency_mask


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=1.0,
                                       channel_axis=-1 if a.ndim == 3 else None))


@dataclass
class MotionField:
    """Flow + trust mask for each consecutive pair (t, t+1) of the reference."""
    fwd: list[np.ndarray]
    masks: list[np.ndarray]

    @property
    def coverage(self) -> float:
        return float(np.mean([m.mean() for m in self.masks]))


def reference_motion(ref: np.ndarray, method: str = "dis",
                     alpha1: float = 0.01, alpha2: float = 0.5) -> MotionField:
    fwd, masks = [], []
    for t in range(len(ref) - 1):
        f = estimate_flow(ref[t], ref[t + 1], method)
        b = estimate_flow(ref[t + 1], ref[t], method)
        fwd.append(f)
        masks.append(fb_consistency_mask(f, b, alpha1, alpha2))
    return MotionField(fwd, masks)


def warp_error_curve(video: np.ndarray, motion: MotionField) -> np.ndarray:
    """E_t = masked mean |V_t - warp(V_{t+1}, F_ref_{t->t+1})|, one value per pair.

    Units are the same as pixel intensity (0..1), averaged over trusted pixels.
    """
    out = []
    for t, (f, m) in enumerate(zip(motion.fwd, motion.masks)):
        w, _ = backward_warp(video[t + 1], f)
        d = np.abs(video[t].astype(np.float32) - w)
        if d.ndim == 3:
            d = d.mean(axis=-1)
        out.append(float(d[m].mean()) if m.any() else np.nan)
    return np.asarray(out)


def per_frame(fn, ref: np.ndarray, rec: np.ndarray) -> np.ndarray:
    return np.asarray([fn(r, x) for r, x in zip(ref, rec)])


@dataclass
class Report:
    name: str
    psnr_mean: float
    ssim_mean: float
    warp_mean: float          # raw masked warping error
    warp_rel: float           # warp_mean / reference floor (1.0 = as consistent as the source)
    warp_curve: np.ndarray
    psnr_curve: np.ndarray


def evaluate(name: str, ref: np.ndarray, rec: np.ndarray, motion: MotionField,
             floor: float) -> Report:
    p = per_frame(psnr, ref, rec)
    s = per_frame(ssim, ref, rec)
    w = warp_error_curve(rec, motion)
    return Report(name, float(np.mean(p)), float(np.mean(s)), float(np.nanmean(w)),
                  float(np.nanmean(w) / floor), w, p)
