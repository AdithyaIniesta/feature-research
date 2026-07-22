#!/usr/bin/env python3
"""Analyze a UAV123 recording from uav_record.py: compare ResNet layers over the sequence.

For the recorded track it computes, per frame, the cosine similarity of each block
(block1..4) against the TEMPLATE, for both the CSRT crop and the ground-truth crop. It
also plots the target's size over time and the CSRT-vs-GT IoU, so you can see which layer
stays most stable as the UAV changes scale, and whether tracker drift (not scale) is what
actually breaks the features.

Usage:
  python3 uav/uav_analyze.py --file results/uav/<seq>/record_<ts>.npz
  python3 uav/uav_analyze.py --seq <seq>          # uses newest recording for that seq

Output (next to the .npz): layer_compare.png, layer_compare.csv
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import cosine   # noqa: E402
from common.geom import xywh_to_xyxy, iou  # noqa: E402

BLOCKS = ["block1", "block2", "block3", "block4"]


def resolve(args):
    if args.file:
        return args.file
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "uav", args.seq)
    files = sorted(glob.glob(os.path.join(base, "record_*.npz")))
    if not files:
        print("No recordings for seq=%s" % args.seq)
        return None
    return files[-1]


def sim_series(data, src, block):
    tmpl = data["template_" + block]
    feats = data[src + "_" + block]
    out = []
    for row in feats:
        out.append(np.nan if np.isnan(row).any() else cosine(tmpl, row))
    return np.array(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--seq")
    args = ap.parse_args()
    if not args.file and not args.seq:
        ap.error("give --file or --seq")

    path = resolve(args)
    if not path:
        return 1
    data = np.load(path)
    frames = data["frame"]
    print("Loaded %s (%d frames)" % (os.path.basename(path), len(frames)))

    # cosine-vs-template series per block, for GT crop (clean) and CSRT crop (drift)
    series = {}
    for src in ("gt", "csrt"):
        for b in BLOCKS:
            series[(src, b)] = sim_series(data, src, b)

    # target size (GT area, normalized to first valid) and CSRT-vs-GT IoU
    gt = data["gt_bbox"]
    csrt = data["csrt_bbox"]
    area = gt[:, 2] * gt[:, 3]
    valid = ~np.isnan(area)
    norm_area = np.full_like(area, np.nan)
    if valid.any():
        norm_area[valid] = area[valid] / area[valid][0]
    ious = []
    for a, b in zip(csrt, gt):
        if np.isnan(a).any() or np.isnan(b).any():
            ious.append(np.nan)
        else:
            ious.append(iou(xywh_to_xyxy(a), xywh_to_xyxy(b)))
    ious = np.array(ious)

    # --- CSV ---
    out_dir = os.path.dirname(path)
    csv_path = os.path.join(out_dir, "layer_compare.csv")
    with open(csv_path, "w") as f:
        hdr = ["frame", "gt_norm_area", "csrt_gt_iou"] \
            + ["gt_sim_" + b for b in BLOCKS] + ["csrt_sim_" + b for b in BLOCKS]
        f.write(",".join(hdr) + "\n")
        for i in range(len(frames)):
            row = [frames[i], norm_area[i], ious[i]] \
                + [series[("gt", b)][i] for b in BLOCKS] \
                + [series[("csrt", b)][i] for b in BLOCKS]
            f.write(",".join("%.4f" % v if isinstance(v, float) or np.isreal(v) else str(v)
                             for v in row) + "\n")
    print("Wrote", csv_path)

    # --- verdict: which block most stable on the clean (GT) crop ---
    print("\nLayer stability vs template on GROUND-TRUTH crop (higher mean, lower std = better):")
    rank = []
    for b in BLOCKS:
        s = series[("gt", b)]
        s = s[~np.isnan(s)]
        rank.append((b, float(s.mean()) if s.size else 0.0,
                     float(s.std()) if s.size else 0.0))
    for b, m, sd in sorted(rank, key=lambda x: -x[1]):
        print("  %-7s mean=%.4f  std=%.4f" % (b, m, sd))

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for b in BLOCKS:
            ax[0].plot(frames, series[("gt", b)], label=b)
        ax[0].set_ylabel("cosine vs template (GT crop)")
        ax[0].set_title("Layer feature stability over sequence")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)

        ax2 = ax[1]
        ax2.plot(frames, norm_area, color="purple", label="target size (norm area)")
        ax2.set_ylabel("target size x")
        ax2.set_xlabel("frame")
        ax2b = ax2.twinx()
        ax2b.plot(frames, ious, color="gray", alpha=0.6, label="CSRT-GT IoU")
        ax2b.set_ylabel("IoU")
        ax2.grid(True, alpha=0.3)
        lines = ax2.get_lines() + ax2b.get_lines()
        ax2.legend(lines, [l.get_label() for l in lines], loc="best")

        png = os.path.join(out_dir, "layer_compare.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print("Wrote", png)
    except Exception as e:
        print("Plot skipped:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
