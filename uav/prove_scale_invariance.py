#!/usr/bin/env python3
"""Controlled proof of scale invariance (isolates scale; no viewpoint/pose confound).

Real footage entangles scale with viewpoint, so it cannot PROVE scale invariance. This
does the clean, controlled test:

  For a target at (cx,cy) in a real frame, we resize the WHOLE frame by a factor s and
  crop a FIXED-size window around the (scaled) target center. The target then appears at
  s x its pixel size inside a constant-size input, with real surrounding context, and
  NOTHING ELSE CHANGES -- same scene, same object, same pose. We measure per-block cosine
  similarity of the features vs the s=1 reference, averaged over many target samples.

If early/mid blocks stay ~1.0 across a wide range of s while block4 falls, that is a
direct proof that those ResNet18 features are scale invariant.

NOTE: unlike the old exp02, we do NOT rescale a fixed 64x64 crop (that only tests
resampling blur). Here the object's pixel FRACTION of the input truly changes with s.

Usage:
  python3 uav/prove_scale_invariance.py --seqs boat6 uav5 car16_1
  python3 uav/prove_scale_invariance.py --seqs boat6 --window 192 --samples 40
Output: results/scale_proof/scale_invariance.csv + .png
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import ResNet18Blocks, cosine   # noqa: E402

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"
BLOCKS = ["block1", "block2", "block3", "block4"]
SCALES = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]


def load_gt(seq):
    p = os.path.join(ANNO_BASE, seq + ".txt")
    if not os.path.isfile(p):
        return None
    out = []
    for line in open(p):
        line = line.strip().replace(",", " ")
        if not line:
            out.append(None); continue
        try:
            x, y, w, h = [float(v) for v in line.split()[:4]]
            out.append(None if (any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0)
                       else [x, y, w, h])
        except Exception:
            out.append(None)
    return out


def window_at_scale(img, cx, cy, s, W):
    """Resize whole image by s, crop a WxW window centered on the scaled target center."""
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(img, (max(1, int(img.shape[1] * s)),
                              max(1, int(img.shape[0] * s))), interpolation=interp)
    ncx, ncy = cx * s, cy * s
    half = W / 2.0
    xi, yi = int(np.floor(ncx - half)), int(np.floor(ncy - half))
    xj, yj = xi + W, yi + W
    pl, pt = max(0, -xi), max(0, -yi)
    pr, pb = max(0, xj - scaled.shape[1]), max(0, yj - scaled.shape[0])
    sub = scaled[max(0, yi):min(scaled.shape[0], yj),
                 max(0, xi):min(scaled.shape[1], xj)]
    if sub.size == 0:
        return None
    if pl or pt or pr or pb:
        sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    if sub.shape[0] != W or sub.shape[1] != W:
        sub = cv2.resize(sub, (W, W))
    return sub


def sample_frames(gt, n, min_size):
    idx = [i for i, b in enumerate(gt) if b and min(b[2], b[3]) >= min_size]
    if not idx:
        return []
    if len(idx) > n:
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, n).astype(int)]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--samples", type=int, default=30, help="target samples per sequence")
    ap.add_argument("--min-size", type=int, default=20)
    args = ap.parse_args()

    ext = ResNet18Blocks(device="cpu")
    W = args.window
    sims = {b: {s: [] for s in SCALES} for b in BLOCKS}
    total = 0

    for seq in args.seqs:
        seq_dir = os.path.join(DATA_BASE, seq)
        frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
            sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        gt = load_gt(seq)
        if not frames or gt is None:
            print("skip %s (missing data)" % seq); continue
        idx = sample_frames(gt, args.samples, args.min_size)
        print("[%s] %d target samples" % (seq, len(idx)))
        for i in idx:
            img = cv2.imread(frames[i])
            if img is None:
                continue
            b = gt[i]
            cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            ref_win = window_at_scale(img, cx, cy, 1.0, W)
            if ref_win is None:
                continue
            ref = ext.block_vectors(ref_win)
            for s in SCALES:
                win = ref_win if s == 1.0 else window_at_scale(img, cx, cy, s, W)
                if win is None:
                    continue
                v = ext.block_vectors(win)
                for blk in BLOCKS:
                    sims[blk][s].append(cosine(ref[blk], v[blk]))
            total += 1

    if total == 0:
        print("No samples processed."); return 1
    print("Total target samples: %d" % total)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "scale_proof")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "scale_invariance.csv")
    with open(csv_path, "w") as f:
        f.write("block,scale,mean_cosine,std_cosine,n\n")
        for blk in BLOCKS:
            for s in SCALES:
                a = np.array(sims[blk][s])
                if a.size:
                    f.write("%s,%.2f,%.4f,%.4f,%d\n"
                            % (blk, s, a.mean(), a.std(), a.size))
    print("Wrote", csv_path)

    print("\nSCALE-INVARIANCE PROOF (cosine vs 1.0x; 1.0 = perfectly invariant):")
    print("  %-7s %8s %8s %8s   %s" % ("block", "@0.25x", "@0.5x", "@2x", "@4x"))
    for blk in BLOCKS:
        def m(s):
            a = sims[blk][s]
            return np.mean(a) if a else float("nan")
        print("  %-7s %8.3f %8.3f %8.3f %8.3f"
              % (blk, m(0.25), m(0.5), m(2.0), m(4.0)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for blk in BLOCKS:
            means = [np.mean(sims[blk][s]) if sims[blk][s] else np.nan for s in SCALES]
            stds = [np.std(sims[blk][s]) if sims[blk][s] else 0 for s in SCALES]
            plt.errorbar(SCALES, means, yerr=stds, marker="o", capsize=3, label=blk)
        plt.axvline(1.0, color="gray", ls="--", lw=1)
        plt.xscale("log")
        plt.xlabel("object scale in fixed input window (log)")
        plt.ylabel("cosine similarity vs 1.0x reference")
        plt.title("Controlled scale-invariance proof (n=%d, %s)"
                  % (total, ",".join(args.seqs)))
        plt.legend(); plt.grid(True, alpha=0.3, which="both")
        png = os.path.join(out_dir, "scale_invariance.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print("Wrote", png)
    except Exception as e:
        print("Plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
