#!/usr/bin/env python3
"""Real-data scale-response curve (the exp02 curve, but from a UAV123 recording).

Every recorded frame has a ground-truth size. We express it as a LINEAR scale factor
relative to the template frame (sqrt(area_i / area_template)), bin the frames by that
factor, and plot the mean block1..4 similarity-to-template per bin. This is directly
comparable to exp02's synthetic-rescale curve, and it avoids the signed-correlation
ambiguity: it shows how each layer's similarity falls as the target's TRUE size departs
from the template size.

Usage:
  python3 uav/uav_scale_curve.py --seq boat6
  python3 uav/uav_scale_curve.py --file results/uav/boat6/record_<ts>.npz
Output (next to the .npz): scale_curve.png, scale_curve.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import cosine   # noqa: E402

BLOCKS = ["block1", "block2", "block3", "block4"]


def resolve(args):
    if args.file:
        return args.file
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "uav", args.seq)
    files = sorted(glob.glob(os.path.join(base, "record_*.npz")))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--seq")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    if not args.file and not args.seq:
        ap.error("give --file or --seq")

    path = resolve(args)
    if not path:
        print("No recording found."); return 1
    data = np.load(path)
    meta_path = path.replace(".npz", ".json")
    tmpl_area = None
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            b = json.load(f).get("template", {}).get("bbox")
            if b:
                tmpl_area = b[2] * b[3]

    gt = data["gt_bbox"]
    area = gt[:, 2] * gt[:, 3]
    if tmpl_area is None:
        valid0 = area[~np.isnan(area)]
        tmpl_area = valid0[0] if valid0.size else 1.0

    lin_scale = np.sqrt(area / tmpl_area)   # linear size factor vs template

    # cosine-vs-template per block on the GT crop
    sim = {}
    for bk in BLOCKS:
        tmpl = data["template_" + bk]
        feats = data["gt_" + bk]
        s = np.array([np.nan if np.isnan(r).any() else cosine(tmpl, r) for r in feats])
        sim[bk] = s

    ok = ~np.isnan(lin_scale) & ~np.isnan(sim["block1"])
    if ok.sum() < args.bins:
        print("Not enough valid frames."); return 1
    ls = lin_scale[ok]

    # log-spaced bins across the observed scale range
    lo, hi = np.percentile(ls, 1), np.percentile(ls, 99)
    edges = np.logspace(np.log10(max(lo, 1e-3)), np.log10(max(hi, lo * 1.01)),
                        args.bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    out_dir = os.path.dirname(path)
    csv_path = os.path.join(out_dir, "scale_curve.csv")
    curve = {bk: [] for bk in BLOCKS}
    counts = []
    with open(csv_path, "w") as f:
        f.write("scale_bin_center,n," + ",".join("mean_" + b for b in BLOCKS) + "\n")
        for i in range(args.bins):
            m = ok & (lin_scale >= edges[i]) & (lin_scale < edges[i + 1])
            n = int(m.sum())
            counts.append(n)
            row = [centers[i], n]
            for bk in BLOCKS:
                v = np.nanmean(sim[bk][m]) if n else np.nan
                curve[bk].append(v)
                row.append(v)
            f.write(",".join("%.4f" % x if isinstance(x, float) else str(x)
                             for x in row) + "\n")
    print("Wrote", csv_path)
    print("\nScale range observed: %.2fx to %.2fx linear (template = 1.0x)"
          % (ls.min(), ls.max()))
    print("Mean similarity at the most extreme scale bin vs near-1.0x:")
    for bk in BLOCKS:
        vals = [v for v in curve[bk] if not np.isnan(v)]
        if vals:
            print("  %-7s  near-template=%.3f   extreme=%.3f   drop=%.3f"
                  % (bk, vals[0], vals[-1], vals[0] - vals[-1]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for bk in BLOCKS:
            plt.plot(centers, curve[bk], marker="o", label=bk)
        plt.axvline(1.0, color="gray", ls="--", lw=1)
        plt.xscale("log")
        plt.xlabel("target linear scale vs template (log)")
        plt.ylabel("mean cosine similarity vs template")
        plt.title("Real-data scale response (%s)"
                  % json.load(open(meta_path)).get("sequence", "")
                  if os.path.isfile(meta_path) else "Real-data scale response")
        plt.legend(); plt.grid(True, alpha=0.3, which="both")
        png = os.path.join(out_dir, "scale_curve.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print("Wrote", png)
    except Exception as e:
        print("Plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
