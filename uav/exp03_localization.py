#!/usr/bin/env python3
"""Experiment 3: feature-correlation vs pixel-correlation LOCALIZATION under scale.

Question: as the target changes size, do deep features FIND its center better than a
classic pixel template? (Not just "stay similar" -- actually localize.)

Method (no training, SiamFC-style single-scale correlation):
  * Template = ground-truth crop at the template frame, normalized to a fixed size.
  * For every later frame we cut a search region centered on the GT center, sized k x the
    TEMPLATE size (so as the target grows past template size, the pattern scale departs --
    exactly the scale stress we want).
  * Pixel:   cv2.matchTemplate NCC of the template over the search region.
  * Feature: normalized cross-correlation of block feature maps (template vs search).
  * Peak -> predicted center. Error = |pred - gt_center| / template_linear_size.
Lower normalized error = better localization. We bin by true scale to see who degrades
faster.

Usage:
  python3 uav/exp03_localization.py --seq boat6
  python3 uav/exp03_localization.py --seq boat6 --blocks block2 block3 --search 3
Output: results/uav/<seq>/exp03_localization.csv + .png
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import ResNet18Blocks   # noqa: E402

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"
CANVAS = 192          # search canvas fed to the net (multiple of 32)


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


def extract_region(img, cx, cy, side):
    """Square region of `side` px centered at (cx,cy), replicate-padded. Returns (sub, x1, y1)."""
    half = side / 2.0
    x1, y1 = cx - half, cy - half
    xi, yi = int(np.floor(x1)), int(np.floor(y1))
    xj, yj = xi + int(round(side)), yi + int(round(side))
    pl, pt = max(0, -xi), max(0, -yi)
    pr, pb = max(0, xj - img.shape[1]), max(0, yj - img.shape[0])
    sub = img[max(0, yi):min(img.shape[0], yj), max(0, xi):min(img.shape[1], xj)]
    if pl or pt or pr or pb:
        sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return sub, x1, y1


def feat_ncc_peak(ext, tmpl_crop, search_img, block, tc):
    """Normalized cross-correlation of block feature maps. Returns (cx,cy) in canvas coords."""
    tm = ext.maps_at(tmpl_crop, tc)[block]        # (1,C,ht,wt)
    sm = ext.maps_at(search_img, CANVAS)[block]   # (1,C,H,W)
    C, ht, wt = tm.shape[1], tm.shape[2], tm.shape[3]
    tn = tm / (tm.norm() + 1e-8)
    num = F.conv2d(sm, tn)                                    # (1,1,H-ht+1,W-wt+1)
    ones = torch.ones((1, C, ht, wt), device=sm.device)
    energy = F.conv2d(sm * sm, ones).clamp(min=1e-8).sqrt()
    resp = (num / energy)[0, 0]
    pi, pj = np.unravel_index(int(resp.argmax().cpu()), resp.shape)
    W = sm.shape[3]
    stride = CANVAS / float(W)
    return (pj + wt / 2.0) * stride, (pi + ht / 2.0) * stride


def pixel_ncc_peak(tmpl_gray, search_gray):
    r = cv2.matchTemplate(search_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, maxloc = cv2.minMaxLoc(r)
    th, tw = tmpl_gray.shape
    return maxloc[0] + tw / 2.0, maxloc[1] + th / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--template-frame", type=int, default=None)
    ap.add_argument("--search", type=float, default=3.0, help="search region = k x template size")
    ap.add_argument("--blocks", nargs="+", default=["block2", "block3"])
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame")
    args = ap.parse_args()

    k = args.search
    tc = int(round(CANVAS / k / 32.0)) * 32
    tc = max(32, tc)                     # template-in-canvas size, multiple of 32
    seq_dir = os.path.join(DATA_BASE, args.seq)
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    gt = load_gt(args.seq)
    if not frames or gt is None:
        print("ERROR: missing frames or GT for %s" % args.seq); return 1

    tf = args.template_frame
    if tf is None:
        tf = next((i for i in range(min(len(frames), len(gt))) if gt[i]), None)
    if tf is None:
        print("ERROR: no valid template GT"); return 1

    ext = ResNet18Blocks(device="cpu")
    timg = cv2.imread(frames[tf])
    tx, ty, tw, th = gt[tf]
    tmpl_crop = timg[int(ty):int(ty + th), int(tx):int(tx + tw)]
    if tmpl_crop.size == 0:
        print("ERROR: empty template crop"); return 1
    tmpl_small = cv2.resize(tmpl_crop, (tc, tc))
    tmpl_gray = cv2.cvtColor(tmpl_small, cv2.COLOR_BGR2GRAY)
    tmpl_area = tw * th
    tmpl_lin = (tw * th) ** 0.5
    region_side = k * max(tw, th)
    methods = ["pixel"] + list(args.blocks)
    print("[exp03] %s: template frame %d, tc=%d, search=%.1fx, methods=%s"
          % (args.seq, tf, tc, k, methods))

    rows = []
    for i in range(tf + 1, len(frames), args.step):
        if i >= len(gt) or gt[i] is None:
            continue
        img = cv2.imread(frames[i])
        if img is None:
            continue
        gx, gy = gt[i][0] + gt[i][2] / 2.0, gt[i][1] + gt[i][3] / 2.0
        region, rx1, ry1 = extract_region(img, gx, gy, region_side)
        region_c = cv2.resize(region, (CANVAS, CANVAS))
        r2img = region_side / float(CANVAS)      # canvas px -> image px

        scale = (gt[i][2] * gt[i][3] / tmpl_area) ** 0.5
        rec = {"frame": i, "scale": scale}
        # pixel
        sg = cv2.cvtColor(region_c, cv2.COLOR_BGR2GRAY)
        cxp, cyp = pixel_ncc_peak(tmpl_gray, sg)
        pred_x, pred_y = rx1 + cxp * r2img, ry1 + cyp * r2img
        rec["pixel"] = float(np.hypot(pred_x - gx, pred_y - gy) / tmpl_lin)
        # features
        for b in args.blocks:
            cxf, cyf = feat_ncc_peak(ext, tmpl_small, region_c, b, tc)
            px, py = rx1 + cxf * r2img, ry1 + cyf * r2img
            rec[b] = float(np.hypot(px - gx, py - gy) / tmpl_lin)
        rows.append(rec)
        if len(rows) % 50 == 0:
            print("  %d frames..." % len(rows))

    if not rows:
        print("No frames processed."); return 1

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "uav", args.seq)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exp03_localization.csv")
    with open(csv_path, "w") as f:
        f.write("frame,scale," + ",".join("err_" + m for m in methods) + "\n")
        for r in rows:
            f.write("%d,%.4f," % (r["frame"], r["scale"])
                    + ",".join("%.4f" % r[m] for m in methods) + "\n")
    print("Wrote", csv_path)

    # summary: overall mean err + success rate (err < 0.5 template)
    print("\nLocalization error (normalized by template size; LOWER = better):")
    for m in methods:
        e = np.array([r[m] for r in rows])
        print("  %-8s mean=%.3f  median=%.3f  success<0.5=%.0f%%"
              % (m, e.mean(), np.median(e), 100.0 * (e < 0.5).mean()))

    # scale-stratified: this is the decisive comparison -- who wins WHERE
    sc = np.array([r["scale"] for r in rows])
    regimes = [("near-scale (0.8-1.5x)", (sc >= 0.8) & (sc < 1.5)),
               ("mid-scale (1.5-2.5x)", (sc >= 1.5) & (sc < 2.5)),
               ("high-scale (>=2.5x)", sc >= 2.5)]
    print("\nMedian error by scale regime (LOWER = better; n = frames in regime):")
    print("  %-22s %6s " % ("regime", "n") + " ".join("%8s" % m for m in methods))
    for name, mask in regimes:
        n = int(mask.sum())
        if n == 0:
            continue
        meds = [np.median(np.array([r[m] for r in rows])[mask]) for m in methods]
        print("  %-22s %6d " % (name, n) + " ".join("%8.3f" % v for v in meds))

    # binned by scale
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc = np.array([r["scale"] for r in rows])
        edges = np.logspace(np.log10(max(sc.min(), 1e-2)),
                            np.log10(max(sc.max(), sc.min() * 1.01)), 9)
        centers = np.sqrt(edges[:-1] * edges[1:])
        plt.figure(figsize=(8, 5))
        for m in methods:
            e = np.array([r[m] for r in rows])
            means = []
            for j in range(len(edges) - 1):
                mask = (sc >= edges[j]) & (sc < edges[j + 1])
                means.append(e[mask].mean() if mask.any() else np.nan)
            plt.plot(centers, means, marker="o", label=m)
        plt.axvline(1.0, color="gray", ls="--", lw=1)
        plt.xscale("log")
        plt.xlabel("target linear scale vs template (log)")
        plt.ylabel("localization error / template size")
        plt.title("Localization vs scale: feature vs pixel (%s)" % args.seq)
        plt.legend(); plt.grid(True, alpha=0.3, which="both")
        png = os.path.join(out_dir, "exp03_localization.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print("Wrote", png)
    except Exception as e:
        print("Plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
