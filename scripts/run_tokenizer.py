"""Run Atlas on a REAL video tokenizer: encode -> decode one real clip, then measure.

What it reports (results/tokenizer_<backend>/):
  summary.json   PSNR / SSIM / warping error of the tokenizer reconstruction,
                 the same for a PSNR-matched blur control, and the seam test
  curves.png     per-pair warping error (reference floor, tokenizer, blur control)
                 with latent-chunk boundaries marked; per-frame PSNR underneath
  frames.png     reference / reconstruction / temporal residual for one pair

Backends
  wan21      Wan 2.1 VAE via diffusers (the tokenizer Cosmos-Predict2, and so
             AnyView, is built on; verify against your AnyView checkout).
             pip install diffusers transformers accelerate
  cosmos-jit NVIDIA Cosmos-0.1 causal video tokenizer, TorchScript checkpoints:
             huggingface-cli download nvidia/Cosmos-0.1-Tokenizer-CV4x8x8 \
                 --local-dir ckpts/CV4x8x8
             then --ckpt ckpts/CV4x8x8
  selftest   no network, no GPU: a fake tokenizer that degrades each 4-frame
             chunk slightly differently. Use it to check the plumbing; its
             numbers mean nothing about any real model.

Examples (RTX 4070, 12 GB):
  python scripts/run_tokenizer.py --video staircase.mp4 --backend wan21 --flow raft
  python scripts/run_tokenizer.py --video staircase.mp4 --backend cosmos-jit \
      --ckpt ckpts/CV4x8x8 --flow raft
  python scripts/run_tokenizer.py --video clip.mp4 --backend selftest --flow dis

If you run out of memory: --height 256 --width 384 --frames 25, or --tiling (wan21).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas import degrade as D  # noqa: E402
from atlas.metrics import evaluate, reference_motion, warp_error_curve  # noqa: E402
from atlas.warp import backward_warp  # noqa: E402


# ----------------------------------------------------------------- input
def load_clip(path, n_frames, h, w, start=0, stride=1, tf=4):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path!r}")
    frames, i = [], 0
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i >= start and (i - start) % stride == 0:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            fh, fw = f.shape[:2]                      # centre-crop to target aspect
            s = min(fh / h, fw / w)
            ch, cw = int(h * s), int(w * s)
            y0, x0 = (fh - ch) // 2, (fw - cw) // 2
            f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    if len(frames) < tf + 1:
        raise SystemExit(f"only {len(frames)} frames read; need at least {tf + 1}")
    n = 1 + tf * ((len(frames) - 1) // tf)          # causal tokenizers take 1 + tf*k frames
    return np.stack(frames[:n]).astype(np.float32) / 255.0


# ----------------------------------------------------------------- backends
def _to_bcthw(x, torch, device, dtype):
    return (torch.from_numpy(x).permute(3, 0, 1, 2)[None] * 2 - 1).to(device=device, dtype=dtype)


def _from_bcthw(y):
    return ((y[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1) / 2).clip(0, 1).astype(np.float32)


def recon_wan21(x, args):  # pragma: no cover - needs GPU + weights
    import torch
    from diffusers import AutoencoderKLWan

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(dev).eval()
    if args.tiling:
        vae.enable_tiling()
    with torch.no_grad():
        t = _to_bcthw(x, torch, dev, dtype)
        z = vae.encode(t).latent_dist.mode()
        y = vae.decode(z).sample
    return _from_bcthw(y), {"latent_shape": list(z.shape), "model": args.model}


def recon_cosmos_jit(x, args):  # pragma: no cover - needs GPU + weights
    import torch

    if not args.ckpt:
        raise SystemExit("--ckpt <dir with encoder.jit and decoder.jit> is required for cosmos-jit")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    enc = torch.jit.load(os.path.join(args.ckpt, "encoder.jit"), map_location=dev).eval().to(dtype)
    dec = torch.jit.load(os.path.join(args.ckpt, "decoder.jit"), map_location=dev).eval().to(dtype)
    first = lambda o: o[0] if isinstance(o, (tuple, list)) else o  # noqa: E731
    with torch.no_grad():
        z = first(enc(_to_bcthw(x, torch, dev, dtype)))
        y = first(dec(z))
    return _from_bcthw(y), {"latent_shape": list(z.shape), "ckpt": args.ckpt}


def recon_selftest(x, args):
    """Fake tokenizer: mild blur everywhere + a per-CHUNK gain and frozen noise
    pattern. Consistent inside each chunk, jumps at chunk boundaries."""
    rng = np.random.default_rng(0)
    out = D.blur(x, 0.8)
    tf = args.temporal_factor
    chunks = [range(0, 1)] + [range(s, min(s + tf, len(x))) for s in range(1, len(x), tf)]
    for c in chunks:
        gain = 1.0 + rng.uniform(-0.02, 0.02)
        pattern = rng.normal(0, 0.01, x.shape[1:]).astype(np.float32)
        for t in c:
            out[t] = np.clip(out[t] * gain + pattern, 0, 1)
    return out, {"note": "synthetic self-test; numbers say nothing about a real model"}


BACKENDS = {"wan21": recon_wan21, "cosmos-jit": recon_cosmos_jit, "selftest": recon_selftest}


# ----------------------------------------------------------------- analysis
def boundary_pairs(n_pairs, tf):
    """Pair (t -> t+1) crosses a latent-chunk boundary for a causal tokenizer
    whose chunks are [0], [1..tf], [tf+1..2tf], ...  i.e. when t % tf == 0."""
    return np.array([t % tf == 0 for t in range(n_pairs)])


def seam_test(excess, is_b, n_perm=20000, seed=0):
    """Is the per-pair excess warping error higher at chunk boundaries?
    Permutation test on the difference of means (one-sided)."""
    b, i = excess[is_b], excess[~is_b]
    obs = float(np.nanmean(b) - np.nanmean(i))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(is_b)
        hits += (np.nanmean(excess[p]) - np.nanmean(excess[~p])) >= obs
    return {"boundary_mean_excess": float(np.nanmean(b)), "interior_mean_excess": float(np.nanmean(i)),
            "boundary_over_interior": float(np.nanmean(b) / np.nanmean(i)) if np.nanmean(i) > 0 else None,
            "n_boundary_pairs": int(is_b.sum()), "n_interior_pairs": int((~is_b).sum()),
            "p_value_one_sided": (hits + 1) / (n_perm + 1)}


def pick_flow(choice):
    if choice != "auto":
        return choice
    try:
        import torch
        import torchvision  # noqa: F401
        return "raft" if torch.cuda.is_available() else "dis"
    except ImportError:
        return "dis"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="wan21: HF repo id")
    ap.add_argument("--ckpt", help="cosmos-jit: dir with encoder.jit / decoder.jit")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame (more motion per pair)")
    ap.add_argument("--temporal-factor", type=int, default=4)
    ap.add_argument("--flow", default="auto", choices=["auto", "raft", "dis", "farneback"])
    ap.add_argument("--bf16", action="store_true", help="wan21: run VAE in bfloat16")
    ap.add_argument("--tiling", action="store_true", help="wan21: tiled VAE to save memory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.height % 16 or args.width % 16:
        raise SystemExit("--height and --width must be multiples of 16")

    out = args.out or os.path.join("results", f"tokenizer_{args.backend}")
    os.makedirs(out, exist_ok=True)
    flow = pick_flow(args.flow)

    ref = load_clip(args.video, args.frames, args.height, args.width, args.start, args.stride,
                    args.temporal_factor)
    print(f"clip: {ref.shape[0]} frames @ {ref.shape[1]}x{ref.shape[2]}  flow={flow}  backend={args.backend}")

    rec, meta = BACKENDS[args.backend](ref, args)
    if rec.shape != ref.shape:
        raise SystemExit(f"reconstruction shape {rec.shape} != reference {ref.shape}")

    motion = reference_motion(ref, flow)
    floor_curve = warp_error_curve(ref, motion)
    floor = float(np.nanmean(floor_curve))
    r = evaluate(args.backend, ref, rec, motion, floor)

    # consistent control at the tokenizer's own PSNR
    sigma, blur, blur_db = D.match_psnr(D.blur, ref, r.psnr_mean, lo=0.0, hi=8.0)
    b = evaluate("blur", ref, blur, motion, floor)
    matched = abs(blur_db - r.psnr_mean) < 0.1

    is_b = boundary_pairs(len(floor_curve), args.temporal_factor)
    seam = seam_test(r.warp_curve - floor_curve, is_b)

    summary = {
        "video": os.path.basename(args.video), "frames": int(ref.shape[0]),
        "size": [int(ref.shape[1]), int(ref.shape[2])], "stride": args.stride,
        "backend": args.backend, "backend_meta": meta, "flow": flow,
        "mask_coverage": motion.coverage, "reference_floor": floor,
        "tokenizer": {"psnr": r.psnr_mean, "ssim": r.ssim_mean, "warp": r.warp_mean, "warp_rel": r.warp_rel},
        "blur_control": {"sigma": sigma, "psnr": b.psnr_mean, "ssim": b.ssim_mean, "warp": b.warp_mean,
                         "warp_rel": b.warp_rel, "psnr_matched": matched},
        "tokenizer_over_blur_at_matched_psnr": r.warp_mean / b.warp_mean if matched else None,
        "seam_test": seam,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    plot(ref, rec, r, b, floor_curve, is_b, motion, out, args.backend)

    print(json.dumps({k: summary[k] for k in ("tokenizer", "blur_control",
                                              "tokenizer_over_blur_at_matched_psnr", "seam_test")}, indent=2))
    if not matched:
        print("note: blur could not reach the tokenizer's PSNR; ratio not reported")


def plot(ref, rec, r, b, floor_curve, is_b, motion, out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False, height_ratios=[2, 1])
    t = np.arange(len(floor_curve))
    for x in t[is_b]:
        ax[0].axvline(x, color="0.75", ls=":", lw=1)
    ax[0].plot(t, floor_curve, "k-", lw=1.5, label="reference floor")
    ax[0].plot(t, r.warp_curve, "k-^", ms=4, label=f"{name}  ({r.psnr_mean:.1f} dB)")
    ax[0].plot(t, b.warp_curve, "k--o", ms=3.5, mfc="white", label=f"blur control  ({b.psnr_mean:.1f} dB)")
    ax[0].set_ylabel("masked warping error")
    ax[0].set_xlabel("frame pair t → t+1   (dotted = pair crosses a latent-chunk boundary)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"Temporal consistency: {name} vs a PSNR-matched consistent control",
                    loc="left", fontweight="bold")
    ax[1].plot(r.psnr_curve, "k-^", ms=3, label=name)
    ax[1].plot(b.psnr_curve, "k--o", ms=3, mfc="white", label="blur control")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=160)

    # frames: the pair with the largest excess warping error
    k = int(np.nanargmax(r.warp_curve - floor_curve))
    w, _ = backward_warp(rec[k + 1], motion.fwd[k])
    res = np.abs(rec[k] - w).mean(-1)
    res[~motion.masks[k]] = np.nan
    cm = plt.get_cmap("Greys").copy()
    cm.set_bad("0.75")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].imshow(ref[k]), ax[0].set_title(f"reference, frame {k}", loc="left")
    ax[1].imshow(rec[k]), ax[1].set_title(f"{name} reconstruction, frame {k}", loc="left")
    ax[2].imshow(res, cmap=cm, vmin=0, vmax=np.nanpercentile(res, 99))
    ax[2].set_title(f"temporal residual {k}→{k+1} (gray = masked)", loc="left")
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "frames.png"), dpi=140)


if __name__ == "__main__":
    main()
