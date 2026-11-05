"""Backward warping and forward-backward occlusion masking."""
from __future__ import annotations

import numpy as np
import cv2


def backward_warp(img: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample `img` at x + flow(x).

    Given flow F_{t->t+1} and frame t+1, returns frame t+1 pulled back into
    frame t's pixel grid. Backward warping (pull) is used instead of forward
    warping (push) because pull gives every output pixel exactly one value:
    no holes, no collisions where two source pixels land on one target.

    Returns (warped, valid) where `valid` is False where x + flow(x) falls
    outside the image.
    """
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + flow[..., 0], gy + flow[..., 1]
    warped = cv2.remap(img.astype(np.float32), mx, my, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return warped, valid


def fb_consistency_mask(fwd: np.ndarray, bwd: np.ndarray,
                        alpha1: float = 0.01, alpha2: float = 0.5) -> np.ndarray:
    """Forward-backward consistency check (Sundaram et al. 2010; Ruder et al. 2016).

    A pixel is trusted if following the forward flow and then the backward
    flow brings it (nearly) home:
        |F(x) + B(x + F(x))|^2 < alpha1 * (|F(x)|^2 + |B(x + F(x))|^2) + alpha2
    Pixels that fail are occluded/disoccluded or the flow is unreliable there,
    so their warp residual says nothing about the video and is excluded.
    """
    bwd_at_fwd, valid = backward_warp(bwd, fwd)
    lhs = np.sum((fwd + bwd_at_fwd) ** 2, axis=-1)
    rhs = alpha1 * (np.sum(fwd ** 2, axis=-1) + np.sum(bwd_at_fwd ** 2, axis=-1)) + alpha2
    return (lhs < rhs) & valid
