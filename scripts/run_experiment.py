"""Atlas v1 experiment: does PSNR see temporal inconsistency? (It does not.)

Reproduce everything with:
    python scripts/run_experiment.py            # DIS flow, ~1-2 min on a laptop CPU
    python scripts/run_experiment.py --flow raft  # if torch/torchvision are installed

Outputs (results/):
    table.csv         every (scene, degradation, strength) row
    summary.json      headline numbers quoted in the README
    atlas_figure.png  the three-panel figure
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skimage import data  # noqa: E402

from atlas import degrade as D  # noqa: E402
from atlas.flow import estimate_flow, endpoint_error  # noqa: E402
from atlas.metrics import reference_motion, warp_error_curve, evaluate  # noqa: E402
from atlas.synth import make_sequence  # noqa: E402

SCENES = {
    # name: (image, camera pan, zoom, rotation, foreground velocity)
    "astronaut": (data.astronaut, (2.0, 0.6), 0.004, 0.15, (3.5, -1.5)),
    "coffee":    (data.coffee,    (-1.5, 1.0), -0.003, -0.2, (-2.5, 2.0)),
    "chelsea":   (data.chelsea,   (1.0, -1.2), 0.006, 0.3, (4.0, 1.0)),
}
BLUR_SIGMAS = [0.6, 0.9, 1.3, 1.8, 2.5]


def build_scene(name, n_frames, seed):
    fn, pan, zoom, rot, vel = SCENES[name]
    return make_sequence(fn(), n_frames=n_frames, pan=pan, zoom=zoom, rot=rot,
                         fg_vel=vel, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="dis", choices=["dis", "farneback", "raft"])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, summary, keep = [], {"flow": args.flow, "frames": args.frames, "seed": args.seed}, {}
    instrument, ablation, crosscheck, blind = {}, {}, {}, {}

    for scene in SCENES:
        ref, gt, _ = build_scene(scene, args.frames, args.seed)

        # 1. Check the instrument before trusting it: flow accuracy vs ground truth.
        epe = {m: float(np.mean([endpoint_error(estimate_flow(ref[t], ref[t + 1], m), gt[t])
                                 for t in range(len(gt))]))
               for m in (["dis", "farneback"] if args.flow != "raft" else ["raft"])}
        motion = reference_motion(ref, args.flow)
        floor = float(np.nanmean(warp_error_curve(ref, motion)))
        instrument[scene] = {"epe_px": epe, "mask_coverage": motion.coverage,
                             "floor_warp_error": floor}

        # 2. Main experiment: consistent blur vs PSNR-matched inconsistent degradations.
        for sigma in BLUR_SIGMAS:
            blur_rec = D.blur(ref, sigma)
            rb = evaluate("blur", ref, blur_rec, motion, floor)
            rows.append(dict(scene=scene, kind="blur", family="consistent", strength=sigma,
                             psnr=rb.psnr_mean, ssim=rb.ssim_mean, warp=rb.warp_mean,
                             warp_rel=rb.warp_rel, ratio_vs_blur=1.0, target_psnr=rb.psnr_mean))
            for kind, fn in D.INCONSISTENT.items():
                s, rec, db = D.match_psnr(fn, ref, rb.psnr_mean, seed=args.seed)
                if abs(db - rb.psnr_mean) > 0.05:      # could not reach this PSNR
                    continue
                r = evaluate(kind, ref, rec, motion, floor)
                rows.append(dict(scene=scene, kind=kind, family="inconsistent", strength=s,
                                 psnr=r.psnr_mean, ssim=r.ssim_mean, warp=r.warp_mean,
                                 warp_rel=r.warp_rel, ratio_vs_blur=r.warp_mean / rb.warp_mean,
                                 target_psnr=rb.psnr_mean))
                if scene == "astronaut" and sigma == 1.3 and kind == "detail_resample":
                    keep = dict(ref=ref, motion=motion, blur=rb, flick=r, floor=floor,
                                sigma=sigma, flick_rec=rec)

        # 3. Ablation: estimate flow on the RECONSTRUCTION instead of the reference.
        ref_mid = D.blur(ref, 1.3)
        target = D.mean_psnr(ref, ref_mid)
        _, nf, _ = D.match_psnr(D.noise_flicker, ref, target, seed=args.seed)
        ablation[scene] = {}
        cases = {
            "detail_resample@matched": D.match_psnr(D.detail_resample, ref, target, seed=args.seed)[1],
            "gain_flicker@0.25": D.gain_flicker(ref, 0.25, seed=args.seed),
        }
        for label, rec in cases.items():
            mot_self = reference_motion(rec, args.flow)
            ablation[scene][label] = {
                "warp_rel_flow_on_reference": float(np.nanmean(warp_error_curve(rec, motion)) / floor),
                "warp_rel_flow_on_reconstruction": float(np.nanmean(warp_error_curve(rec, mot_self)) / floor),
                "mask_coverage_reference": motion.coverage,
                "mask_coverage_reconstruction": mot_self.coverage,
                "epe_px_reference": float(np.mean([endpoint_error(f, g) for f, g in zip(motion.fwd, gt)])),
                "epe_px_reconstruction": float(np.mean([endpoint_error(f, g) for f, g in zip(mot_self.fwd, gt)])),
            }

        # 4. Cross-check with an independent flow method.
        if args.flow == "dis":
            m2 = reference_motion(ref, "farneback")
            f2 = float(np.nanmean(warp_error_curve(ref, m2)))
            b = evaluate("blur", ref, ref_mid, m2, f2)
            n = evaluate("noise_flicker", ref, nf, m2, f2)
            crosscheck[scene] = {"farneback_ratio": n.warp_mean / b.warp_mean}

        # 5. Blind spot: consistent-but-wrong. Both matched to the blur's PSNR.
        _, bias, _ = D.match_psnr(D.global_bias, ref, target, seed=args.seed)
        _, drift, _ = D.match_psnr(D.gain_drift, ref, target, seed=args.seed)
        blind[scene] = {
            "psnr": target,
            "blur_warp_rel": evaluate("b", ref, ref_mid, motion, floor).warp_rel,
            "global_bias_warp_rel": evaluate("bias", ref, bias, motion, floor).warp_rel,
            "gain_drift_warp_rel": evaluate("drift", ref, drift, motion, floor).warp_rel,
            "noise_flicker_warp_rel": evaluate("nf", ref, nf, motion, floor).warp_rel,
        }

    # ---- write table
    with open(os.path.join(args.out, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    # ---- headline numbers
    inc = [r for r in rows if r["family"] == "inconsistent"]
    by_kind = {}
    for k in D.INCONSISTENT:
        rs = [r["ratio_vs_blur"] for r in inc if r["kind"] == k]
        if rs:
            by_kind[k] = {"median": float(np.median(rs)), "min": float(np.min(rs)),
                          "max": float(np.max(rs)), "n": len(rs)}
    peak = max(inc, key=lambda r: r["ratio_vs_blur"])
    summary.update(
        instrument=instrument,
        ratio_vs_matched_blur=by_kind,
        all_inconsistent=dict(median=float(np.median([r["ratio_vs_blur"] for r in inc])),
                              min=float(np.min([r["ratio_vs_blur"] for r in inc])),
                              max=float(np.max([r["ratio_vs_blur"] for r in inc])), n=len(inc)),
        peak_row=peak,
        consistent_warp_rel_range=[float(min(r["warp_rel"] for r in rows if r["family"] == "consistent")),
                                   float(max(r["warp_rel"] for r in rows if r["family"] == "consistent"))],
        inconsistent_warp_rel_range=[float(min(r["warp_rel"] for r in inc)),
                                     float(max(r["warp_rel"] for r in inc))],
        ablation_flow_on_reconstruction=ablation,
        crosscheck_farneback=crosscheck,
        blind_spot_gain_drift=blind,
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    make_figure(rows, keep, blind, os.path.join(args.out, "atlas_figure.png"))
    print(json.dumps({k: summary[k] for k in ("ratio_vs_matched_blur", "all_inconsistent")}, indent=2))


def make_figure(rows, keep, blind, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))

    # (a) PSNR vs warping error — the two families, marked by shape AND fill (B&W-safe)
    style = {"blur": dict(marker="o", mfc="white", mec="black", label="blur (consistent)"),
             "noise_flicker": dict(marker="^", mfc="black", mec="black", label="noise flicker"),
             "gain_flicker": dict(marker="s", mfc="0.55", mec="black", label="gain flicker"),
             "detail_resample": dict(marker="D", mfc="0.85", mec="black", label="detail re-sampling")}
    for kind, st in style.items():
        rs = [r for r in rows if r["kind"] == kind]
        ax[0].plot([r["psnr"] for r in rs], [r["warp_rel"] for r in rs], ls="none", ms=6, **st)
    ax[0].axhline(1.0, color="black", ls="--", lw=0.8)
    ax[0].text(0.02, 0.30, "reference floor (dashed)", transform=ax[0].transAxes, fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("mean PSNR (dB)  →  'cleaner'")
    ax[0].set_ylabel("warping error / reference floor (log)")
    ax[0].set_title("(a) Same PSNR, very different temporal error")
    ax[0].legend(fontsize=7, frameon=False, loc="upper right")

    # (b) per-frame curves for one PSNR-matched pair
    if keep:
        b, fl = keep["blur"], keep["flick"]
        t = np.arange(len(b.warp_curve))
        ax[1].plot(t, b.warp_curve, "k-o", ms=3, mfc="white", label=f"blur σ={keep['sigma']}  ({b.psnr_mean:.1f} dB)")
        ax[1].plot(t, fl.warp_curve, "k--^", ms=3, label=f"detail re-sampling  ({fl.psnr_mean:.1f} dB)")
        ax[1].axhline(keep["floor"], color="0.4", ls=":", lw=1, label="reference floor")
        ax[1].set_xlabel("frame pair t → t+1")
        ax[1].set_ylabel("masked warping error")
        ax[1].set_title(f"(b) Matched pair: {fl.warp_mean / b.warp_mean:.0f}× the warping error")
        ax[1].legend(fontsize=7, frameon=False)

    # (c) the blind spot — all four bars at the SAME PSNR
    names = list(blind)
    x = np.arange(len(names))
    bars = [("blur_warp_rel", "blur", "white", ""),
            ("global_bias_warp_rel", "uniformly darker", "0.85", ".."),
            ("gain_drift_warp_rel", "slow brightness drift", "0.6", "//"),
            ("noise_flicker_warp_rel", "noise flicker", "0.25", "")]
    wd = 0.2
    for i, (key, lab, col, hatch) in enumerate(bars):
        ax[2].bar(x + (i - 1.5) * wd, [blind[n][key] for n in names], wd, color=col,
                  edgecolor="black", hatch=hatch, label=lab)
    ax[2].axhline(1.0, color="black", ls="--", lw=0.8)
    ax[2].set_xticks(x, names)
    ax[2].set_ylabel("warping error / reference floor")
    ax[2].set_title("(c) Blind spot: stable-but-wrong reads clean")
    ax[2].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
