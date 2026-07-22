#!/usr/bin/env python3
"""Scan UAV123 ground-truth annotations and rank sequences by how much the target's
SCALE and ASPECT RATIO change -- so you can pick sequences that actually test scale
invariance (person1 barely changes size and is a poor scale test).

For each sequence it reports:
  frames      : number of valid (non-occluded) GT boxes
  scale_ratio : max(area) / min(area)      -> how much the target grows/shrinks
  lin_ratio   : sqrt(scale_ratio)          -> linear size change (more intuitive)
  aspect_rng  : max(w/h) / min(w/h)        -> aspect-ratio distortion

Usage:
  python3 uav/uav_scan.py                 # rank all sequences by scale
  python3 uav/uav_scan.py --sort aspect   # rank by aspect-ratio change
  python3 uav/uav_scan.py --top 20
"""
import argparse
import glob
import os

import numpy as np

ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"


def stats_for(path):
    ws, hs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            try:
                x, y, w, h = [float(v) for v in line.split()[:4]]
            except Exception:
                continue
            if any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0:
                continue
            ws.append(w); hs.append(h)
    if len(ws) < 2:
        return None
    ws, hs = np.array(ws), np.array(hs)
    area = ws * hs
    aspect = ws / hs
    scale_ratio = float(area.max() / area.min())
    aspect_rng = float(aspect.max() / aspect.min())
    return {"frames": len(ws), "scale_ratio": scale_ratio,
            "lin_ratio": scale_ratio ** 0.5, "aspect_rng": aspect_rng}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="scale", choices=["scale", "aspect", "frames"])
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ANNO_BASE, "*.txt")))
    if not files:
        print("No annotations under %s" % ANNO_BASE)
        return 1

    rows = []
    for p in files:
        s = stats_for(p)
        if s:
            s["seq"] = os.path.splitext(os.path.basename(p))[0]
            rows.append(s)

    key = {"scale": "scale_ratio", "aspect": "aspect_rng", "frames": "frames"}[args.sort]
    rows.sort(key=lambda r: -r[key])

    print("\n%-18s %7s %11s %10s %11s" %
          ("sequence", "frames", "scale_x", "linear_x", "aspect_x"))
    print("-" * 62)
    for r in rows[:args.top]:
        print("%-18s %7d %11.2f %10.2f %11.2f" %
              (r["seq"], r["frames"], r["scale_ratio"], r["lin_ratio"], r["aspect_rng"]))

    big = [r for r in rows if r["lin_ratio"] >= 2.0]
    print("\n%d sequences have >=2x linear scale change (good scale tests)." % len(big))
    if big:
        print("Top picks:", ", ".join(r["seq"] for r in
                                       sorted(big, key=lambda r: -r["scale_ratio"])[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
