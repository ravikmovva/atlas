"""Synthetic test sequences with ground-truth motion.

A natural image is filmed by a virtual camera that pans, zooms and rotates
slightly, while a foreground patch moves independently across it. Because
the motion is constructed, the ground-truth flow and the occlusion regions
are known exactly. That lets the harness check its own instrument (flow
end-point error) before trusting any number it reports about a video.
"""
from __future__ import annotations

import numpy as np
import cv2


def _affine(tx, ty, scale, rot_deg, cx, cy):
    """2x3 matrix mapping frame coords -> source-image coords."""
    a = np.deg2rad(rot_deg)
    c, s = np.cos(a) / scale, np.sin(a) / scale
    m = np.array([[c, -s, 0.0], [s, c, 0.0]])
    # rotate/scale about the frame centre, then translate
    m[:, 2] = np.array([cx + tx, cy + ty]) - m[:, :2] @ np.array([cx, cy])
    return m


def _invert(m):
    a = np.vstack([m, [0, 0, 1]])
    return np.linalg.inv(a)[:2]


def make_sequence(image: np.ndarray, n_frames: int = 24, size=(192, 256),
                  pan=(2.0, 0.6), zoom=0.004, rot=0.15, fg_vel=(3.5, -1.5),
                  fg_size=48, seed: int = 0):
    """Return (frames [T,H,W,3] float32 in 0..1, gt_flow [T-1,H,W,2], fg_masks [T,H,W])."""
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    H, W = size
    cx, cy = img.shape[1] / 2, img.shape[0] / 2

    # foreground: a patch cut from elsewhere in the image, rotated so it is
    # visually distinct from the background around it
    sy = rng.integers(0, img.shape[0] - fg_size)
    sx = rng.integers(0, img.shape[1] - fg_size)
    patch = np.ascontiguousarray(np.rot90(img[sy:sy + fg_size, sx:sx + fg_size]))
    p0 = np.array([W * 0.25, H * 0.45])

    mats, frames, fg_masks, fg_pos = [], [], [], []
    for t in range(n_frames):
        m = _affine(pan[0] * t - W / 2, pan[1] * t - H / 2, 1.0 + zoom * t, rot * t, W / 2, H / 2)
        m[:, 2] += [cx, cy]
        mats.append(m)
        mi = _invert(m)
        f = cv2.warpAffine(img, mi, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        p = p0 + np.array(fg_vel) * t
        x0, y0 = int(round(p[0])), int(round(p[1]))
        mask = np.zeros((H, W), bool)
        ys, xs = slice(max(y0, 0), min(y0 + fg_size, H)), slice(max(x0, 0), min(x0 + fg_size, W))
        f[ys, xs] = patch[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        mask[ys, xs] = True
        frames.append(f)
        fg_masks.append(mask)
        fg_pos.append((x0, y0))

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([gx, gy, np.ones_like(gx)], -1)
    flows = []
    for t in range(n_frames - 1):
        # background point: frame t -> source image -> frame t+1
        u = pts @ mats[t].T
        nxt = np.concatenate([u, np.ones_like(u[..., :1])], -1) @ _invert(mats[t + 1]).T
        fl = (nxt - pts[..., :2]).astype(np.float32)
        dx = fg_pos[t + 1][0] - fg_pos[t][0]
        dy = fg_pos[t + 1][1] - fg_pos[t][1]
        fl[fg_masks[t]] = (dx, dy)
        flows.append(fl)
    return np.stack(frames), np.stack(flows), np.stack(fg_masks)
