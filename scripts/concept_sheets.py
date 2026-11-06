"""Two concept sheets built from real Atlas data (B&W-print safe)."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skimage import data  # noqa: E402
from atlas import degrade as D  # noqa: E402
from atlas.metrics import reference_motion, warp_error_curve, per_frame, psnr  # noqa: E402
from atlas.synth import make_sequence  # noqa: E402
from atlas.warp import backward_warp  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

ref, gt, fg = make_sequence(data.astronaut(), n_frames=24)
mot = reference_motion(ref)
floor = np.nanmean(warp_error_curve(ref, mot))
blur = D.blur(ref, 1.3)
target = D.mean_psnr(ref, blur)
_, flick, _ = D.match_psnr(D.detail_resample, ref, target)
T = 8


def residual(v, t):
    w, _ = backward_warp(v[t + 1], mot.fwd[t])
    r = np.abs(v[t] - w).mean(-1)
    r[~mot.masks[t]] = np.nan
    return r


def box(ax, xy, w, h, text, size=8.5, fc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01", fc=fc, ec="black", lw=0.8, clip_on=False,
                                transform=ax.transAxes))
    ax.text(xy[0] + 0.015, xy[1] + h - 0.02, text, transform=ax.transAxes, va="top", fontsize=size,
            family="DejaVu Sans", linespacing=1.45)


# ------------------------------------------------------------ Sheet 1
fig = plt.figure(figsize=(14, 9.2))
fig.suptitle("Atlas, step by step: does the reconstruction move the way the scene moves?",
             fontsize=14, fontweight="bold", y=0.985)
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.55], hspace=0.38, wspace=0.12,
                      left=0.03, right=0.97, top=0.92, bottom=0.03)


def show(ax, img, title, cmap=None, **kw):
    ax.imshow(img, cmap=cmap, **kw)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)


gray = lambda x: x.mean(-1)  # noqa: E731
show(fig.add_subplot(gs[0, 0]), gray(ref[T]), "① Reference frame t", "gray", vmin=0, vmax=1)
show(fig.add_subplot(gs[0, 1]), gray(ref[T + 1]), "   Reference frame t+1", "gray", vmin=0, vmax=1)

ax = fig.add_subplot(gs[0, 2])
mag = np.linalg.norm(mot.fwd[T], axis=-1)
show(ax, mag, "② Flow F(t→t+1), measured on the REFERENCE", "Greys", vmin=0, vmax=mag.max())
st = 16
yy, xx = np.mgrid[st // 2:mag.shape[0]:st, st // 2:mag.shape[1]:st]
ax.quiver(xx, yy, mot.fwd[T][yy, xx, 0], mot.fwd[T][yy, xx, 1], color="black", angles="xy",
          scale_units="xy", scale=0.35, width=0.004)
ax.text(4, mag.shape[0] - 6, "dark = fast; arrows = direction\n(camera pans + moving patch)",
        fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.85))

ax = fig.add_subplot(gs[0, 3])
show(ax, mot.masks[T].astype(float), "③ Trust mask (fwd-bwd check)", "gray", vmin=0, vmax=1)
ax.text(4, mag.shape[0] - 6, f"white = trusted ({mot.masks[T].mean():.0%})\n"
        "black = occluded / unreliable:\nedges of the moving patch",
        fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.85))

r_b, r_f = residual(blur, T), residual(flick, T)
vmax = np.nanpercentile(r_f, 99)
cm = plt.get_cmap("Greys").copy()
cm.set_bad("0.75")
show(fig.add_subplot(gs[1, 0]), gray(blur[T]), f"Recon A: blur  —  PSNR {target:.1f} dB", "gray",
     vmin=0, vmax=1)
show(fig.add_subplot(gs[1, 1]), r_b,
     f"④ A: |A(t) − warp(A(t+1))|   mean {np.nanmean(r_b):.4f}", cm, vmin=0, vmax=vmax)
show(fig.add_subplot(gs[1, 2]), gray(flick[T]),
     f"Recon B: detail re-sampling  —  PSNR {D.mean_psnr(ref, flick):.1f} dB", "gray", vmin=0, vmax=1)
show(fig.add_subplot(gs[1, 3]), r_f,
     f"④ B: |B(t) − warp(B(t+1))|   mean {np.nanmean(r_f):.4f}", cm, vmin=0, vmax=vmax)

ax = fig.add_subplot(gs[2, :])
ax.axis("off")
ratio = np.nanmean(warp_error_curve(flick, mot)) / np.nanmean(warp_error_curve(blur, mot))
box(ax, (0.005, 0.08), 0.23, 0.84,
    "① Two frames of the REFERENCE.\nThe camera pans/zooms/rotates;\na patch moves on its own path,\nso some pixels get covered/uncovered.")
box(ax, (0.255, 0.08), 0.23, 0.84,
    "② Optical flow: for each pixel in t,\nwhere did it go in t+1?\nComputed on the reference, so every\nreconstruction is scored on the same\nmotion and the same pixels.")
box(ax, (0.505, 0.08), 0.23, 0.84,
    "③ Go forward along F, come back\nalong B. Pixels that don't return home\nare occlusions or bad flow: excluded.\nGray in ④ = excluded pixels.")
box(ax, (0.755, 0.08), 0.24, 0.84,
    f"④ Pull t+1 back onto t and subtract.\nSame PSNR, same colour scale:\nA's residual is near-blank, B's is lit.\nOver the clip B has {ratio:.0f}× A's error.\nPSNR scores them identically.",
    fc="0.93")
fig.savefig(os.path.join(OUT, "concept_1_how_it_measures.png"), dpi=150)

# ------------------------------------------------------------ Sheet 2
fig = plt.figure(figsize=(14, 9.2))
fig.suptitle("Why PSNR can't see flicker — and what Atlas can't see either",
             fontsize=14, fontweight="bold", y=0.985)
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28, left=0.06, right=0.97, top=0.9, bottom=0.07,
                      width_ratios=[1.15, 1, 1])

# (a) one scene point followed through time
import cv2  # noqa: E402


def bilinear(img, x, y):
    return float(cv2.getRectSubPix(img.mean(-1).astype(np.float32), (1, 1), (float(x), float(y)))[0, 0])


def follow(p):
    tr = [np.array(p, float)]
    for t in range(len(gt)):
        q = tr[-1]
        fx = cv2.getRectSubPix(np.ascontiguousarray(gt[t][..., 0]), (1, 1), (q[0], q[1]))[0, 0]
        fy = cv2.getRectSubPix(np.ascontiguousarray(gt[t][..., 1]), (1, 1), (q[0], q[1]))[0, 0]
        tr.append(q + [fx, fy])
    return tr


# choose a mid-gray, low-gradient background point that stays in frame and off the patch
g0 = cv2.GaussianBlur(ref[0].mean(-1), (0, 0), 1.0)
gm = np.hypot(*np.gradient(g0))
best, track = None, None
for y in range(20, 172, 6):
    for x in range(20, 236, 6):
        if fg[0][y, x] or not (0.3 < g0[y, x] < 0.7):
            continue
        tr = follow((x, y))
        if all(8 < q[0] < 248 and 8 < q[1] < 184 for q in tr) and not any(
                fg[t][int(q[1]), int(q[0])] for t, q in enumerate(tr)):
            if best is None or gm[y, x] < best:
                best, track = gm[y, x], tr


def sample(v):
    return np.array([bilinear(v[t], q[0], q[1]) for t, q in enumerate(track)])


ax = fig.add_subplot(gs[0, 0])
tt = np.arange(len(track))
ax.plot(tt, sample(ref), "k-", lw=2.2, label="reference")
ax.plot(tt, sample(blur), "k--o", ms=3.5, mfc="white", label="A: blur (consistent)")
ax.plot(tt, sample(flick), "k:^", ms=3.5, label="B: detail re-sampling (flicker)")
ax.set_xlabel("frame")
ax.set_ylabel("brightness of ONE scene point")
ax.set_title("(a) Follow one point as it moves", loc="left", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3)
ax.set_ylim(ax.get_ylim()[0] - 0.05, ax.get_ylim()[1] + 0.05)
ax.text(0.98, 0.03, "Reference and A stay steady (the point keeps its colour).\nB jumps every frame. Optical flow is what lets us follow the point.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        bbox=dict(fc="white", ec="none", alpha=0.9))

# (b) per-frame PSNR vs warping error
pb, pf = per_frame(psnr, ref, blur), per_frame(psnr, ref, flick)
wb, wf = warp_error_curve(blur, mot), warp_error_curve(flick, mot)
ax = fig.add_subplot(gs[0, 1])
ax.plot(pb, "k--o", ms=3, mfc="white", label="A")
ax.plot(pf, "k:^", ms=3, label="B")
ax.set_ylim(min(pb.min(), pf.min()) - 3, max(pb.max(), pf.max()) + 3)
ax.set_xlabel("frame")
ax.set_ylabel("PSNR (dB)")
ax.set_title("(b) PSNR: the two lines overlap", loc="left", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)
ax.text(0.03, 0.05, "Each frame scored alone.\nNo term links t to t+1.", transform=ax.transAxes,
        fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.plot(wb, "k--o", ms=3, mfc="white", label="A")
ax.plot(wf, "k:^", ms=3, label="B")
ax.axhline(floor, color="0.45", ls="-", lw=1)
ax.text(0.5, floor * 1.25, "reference floor", fontsize=7.5, color="0.3")
ax.set_yscale("log")
ax.set_xlabel("frame pair t → t+1")
ax.set_ylabel("masked warping error (log)")
ax.set_title(f"(c) Atlas: {ratio:.0f}× apart", loc="left", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="center right")

# (d) the 2x2: correct per frame vs consistent over time
ax = fig.add_subplot(gs[1, 0:2])
ax.set_xlim(0, 2), ax.set_ylim(0, 2)
ax.set_xticks([]), ax.set_yticks([])
ax.axvline(1, color="black", lw=0.8), ax.axhline(1, color="black", lw=0.8)
ax.set_xlabel("each frame close to the reference?   (what PSNR / SSIM / VMAF ask)  →", fontsize=9)
ax.set_ylabel("consistent along motion?\n(what Atlas asks)  →", fontsize=9)
ax.set_title("(d) Two different questions — you need both", loc="left", fontweight="bold")
q = dict(fontsize=8.5, ha="center", va="center", linespacing=1.4, bbox=dict(fc="white", ec="none", pad=4))
ax.text(1.5, 1.5, "GOOD\nreference itself;\na faithful reconstruction", **q)
ax.text(0.5, 1.5, "STABLE BUT WRONG\nuniformly darker clip, blur\nPSNR flags it · Atlas: clean\n(Atlas's blind spot)", **q)
ax.text(1.5, 0.5, "RIGHT-ISH BUT FLICKERS\ndetail re-sampling, gain jitter\nPSNR: fine · Atlas: flags it\n(PSNR's blind spot — the finding)", **q)
ax.text(0.5, 0.5, "BAD\nboth metrics flag it", **q)
ax.add_patch(plt.Rectangle((1, 0), 1, 1, fc="0.9", ec="none", zorder=0))
ax.add_patch(plt.Rectangle((0, 1), 1, 1, fc="white", ec="black", hatch="..", lw=0, zorder=0, alpha=0.2))

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
box(ax, (0.0, 0.0), 1.0, 1.0,
    "The measurement in one line\n\n"
    "  E_t = mean over trusted pixels of\n"
    "        | X(t) − X(t+1) pulled back along F |\n\n"
    "F  = flow from the reference\n"
    "trusted = passes forward-backward check\n"
    "floor = same E on the reference itself\n\n"
    "Headline (45 PSNR-matched pairs, 3 scenes)\n"
    "  flicker / consistent: 2.7× – 44×, median 13×\n"
    "  ratio grows as quality drops: quote it\n"
    "  with its operating point\n\n"
    "Why TRI cares\n"
    "  view synthesis / 4D: per-frame metrics and\n"
    "  FVD don't score per-sequence consistency", size=8.6, fc="0.96")
fig.savefig(os.path.join(OUT, "concept_2_why_psnr_is_blind.png"), dpi=150)
print("ok", ratio)
