"""Optical flow estimation with pluggable backends.

Convention: flow[y, x] = (dx, dy) such that pixel (x, y) in frame A
moves to (x + dx, y + dy) in frame B.

Backends
--------
dis        OpenCV DIS (fast, CPU, default)
farneback  OpenCV Farneback (classical, used as a cross-check)
raft       torchvision RAFT-Large (optional; needs torch + weights, GPU recommended)

Using two independent estimators is deliberate: if a finding only shows
up under one flow method, it is a property of the flow method, not of
the video being measured.
"""
from __future__ import annotations

import numpy as np
import cv2


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def flow_dis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(_to_gray_u8(a), _to_gray_u8(b), None).astype(np.float32)


def flow_farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        _to_gray_u8(a), _to_gray_u8(b), None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0,
    ).astype(np.float32)


_RAFT = None


def flow_raft(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pragma: no cover - optional
    """RAFT-Large via torchvision. Install torch/torchvision to enable."""
    global _RAFT
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _RAFT is None:
        w = Raft_Large_Weights.DEFAULT
        _RAFT = (raft_large(weights=w).eval().to(dev), w.transforms())
    model, tf = _RAFT

    def prep(x):
        if x.ndim == 2:
            x = np.repeat(x[..., None], 3, axis=2)
        return torch.from_numpy(x).permute(2, 0, 1)[None].float()

    ta, tb = tf(prep(a), prep(b))
    h, w = a.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8  # RAFT needs /8 sizes
    ta = torch.nn.functional.pad(ta, (0, pw, 0, ph), mode="replicate")
    tb = torch.nn.functional.pad(tb, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        f = model(ta.to(dev), tb.to(dev))[-1][0]
    return f[:, :h, :w].permute(1, 2, 0).cpu().numpy().astype(np.float32)


BACKENDS = {"dis": flow_dis, "farneback": flow_farneback, "raft": flow_raft}


def estimate_flow(a: np.ndarray, b: np.ndarray, method: str = "dis") -> np.ndarray:
    try:
        return BACKENDS[method](a, b)
    except KeyError:
        raise ValueError(f"unknown flow method {method!r}; choose from {list(BACKENDS)}")


def endpoint_error(f_est: np.ndarray, f_gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Average end-point error (pixels), the standard flow accuracy measure."""
    e = np.linalg.norm(f_est - f_gt, axis=-1)
    return float(e[mask].mean() if mask is not None else e.mean())
