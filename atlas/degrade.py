"""Controlled degradations: temporally consistent vs temporally inconsistent.

Every degradation is a function (ref video, strength, seed) -> reconstruction.
The experiment compares pairs at MATCHED PSNR, so any difference in the
warping error cannot be explained by "one is just worse than the other".

Consistent (a deterministic function of the content, so it moves with the scene):
    blur           Gaussian blur, same sigma every frame (detail loss)

Inconsistent (each frame is degraded independently):
    noise_flicker  i.i.d. Gaussian noise, a fresh draw every frame
    gain_flicker   global brightness gain jittered frame to frame
    detail_resample blur + freshly sampled band-pass "texture" every frame;
                   the closest stand-in for a generative decoder or tokenizer
                   that re-hallucinates fine detail on each frame

Consistent but wrong (the documented blind spot):
    global_bias    whole clip uniformly darker (perfectly stable, still wrong)
    gain_drift     brightness ramps slowly over the clip
"""
from __future__ import annotations

import numpy as np
import cv2

from .metrics import psnr


def _blur(frame, sigma):
    if sigma <= 0:
        return frame.copy()
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def blur(ref, sigma, seed=0):
    return np.stack([_blur(f, sigma) for f in ref])


def noise_flicker(ref, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(ref + rng.normal(0, sigma, ref.shape).astype(np.float32), 0, 1)


def gain_flicker(ref, amp, seed=0):
    rng = np.random.default_rng(seed)
    g = 1.0 + amp * rng.choice([-1.0, 1.0], size=len(ref))
    return np.clip(ref * g[:, None, None, None].astype(np.float32), 0, 1)


def detail_resample(ref, amp, seed=0, base_sigma=0.5, tex_sigma=0.8):
    rng = np.random.default_rng(seed)
    out = []
    for f in ref:
        n = rng.normal(0, 1, f.shape[:2]).astype(np.float32)
        n = _blur(n, tex_sigma)
        n /= n.std() + 1e-8
        out.append(np.clip(_blur(f, base_sigma) + amp * n[..., None], 0, 1))
    return np.stack(out)


def global_bias(ref, amp, seed=0):
    return np.clip(ref * (1.0 - amp), 0, 1).astype(np.float32)


def gain_drift(ref, amp, seed=0):
    g = 1.0 + amp * np.linspace(-1, 1, len(ref))
    return np.clip(ref * g[:, None, None, None].astype(np.float32), 0, 1)


INCONSISTENT = {"noise_flicker": noise_flicker, "gain_flicker": gain_flicker,
                "detail_resample": detail_resample}


def mean_psnr(ref, rec):
    return float(np.mean([psnr(a, b) for a, b in zip(ref, rec)]))


def match_psnr(fn, ref, target_db, lo=0.0, hi=0.5, seed=0, iters=30, **kw):
    """Bisect the strength of `fn` so the result's mean PSNR equals target_db.

    PSNR decreases monotonically with strength for these degradations.
    Returns (strength, reconstruction, achieved_db).
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_psnr(ref, fn(ref, mid, seed=seed, **kw)) > target_db:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    rec = fn(ref, s, seed=seed, **kw)
    return s, rec, mean_psnr(ref, rec)
