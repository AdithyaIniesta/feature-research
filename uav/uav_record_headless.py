#!/usr/bin/env python3
"""Headless UAV123 recorder -- no display, no clicking. Works over SSH.

Uses the UAV123 ground-truth box as the template (from the first valid frame, or
--template-frame N) and records block1..4 ResNet features of the GT crop for EVERY
frame. Also runs CSRT from that template so you still get the drift comparison.

This is the batch-friendly counterpart to uav_record.py (which needs a monitor for
click-select). For scale/aspect analysis, GT-based recording is actually cleaner.

Usage:
  python3 uav/uav_record_headless.py boat6
  python3 uav/uav_record_headless.py boat6 --template-frame 0 --min-box 12
Output: results/uav/<seq>/record_<ts>.npz  (+ .json), same format as uav_record.py
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import ResNet18Blocks   # noqa: E402
from common.geom import crop                 # noqa: E402

BLOCKS = ["block1", "block2", "block3", "block4"]
DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"


def load_gt(seq_name):
    p = os.path.join(ANNO_BASE, seq_name + ".txt")
    if not os.path.isfile(p):
        return None
    boxes = []
    with open(p) as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                boxes.append(None); continue
            try:
                x, y, w, h = [float(v) for v in line.split()[:4]]
                boxes.append(None if (any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0)
                             else [x, y, w, h])
            except Exception:
                boxes.append(None)
    return boxes


def make_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", help="sequence name, e.g. boat6")
    ap.add_argument("--template-frame", type=int, default=None,
                    help="frame index for template (default: first valid GT frame)")
    ap.add_argument("--min-box", type=int, default=8,
                    help="skip feature extraction when GT box smaller than this (px)")
    args = ap.parse_args()

    seq_dir = os.path.join(DATA_BASE, args.seq)
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if not frames:
        print("ERROR: no images in %s" % seq_dir)
        return 1
    gt = load_gt(args.seq)
    if gt is None:
        print("ERROR: no GT annotation for %s (headless needs GT)" % args.seq)
        return 1
    print("[SEQ] %s: %d frames, %d gt lines" % (args.seq, len(frames), len(gt)))

    # pick template frame
    tf = args.template_frame
    if tf is None:
        tf = next((i for i in range(min(len(frames), len(gt))) if gt[i]), None)
    if tf is None or tf >= len(gt) or gt[tf] is None:
        print("ERROR: no valid GT box for template frame")
        return 1

    ext = ResNet18Blocks(device="cpu")
    tframe = cv2.imread(frames[tf])
    tcrop = crop(tframe, gt[tf], pad=0.15)
    template_vec = ext.block_vectors(tcrop)
    tracker = make_csrt()
    tracker.init(tframe, tuple(int(v) for v in gt[tf]))
    print("[TEMPLATE] frame %d bbox %s" % (tf, [int(v) for v in gt[tf]]))

    rec = {"frame": [], "csrt_bbox": [], "gt_bbox": []}
    for b in BLOCKS:
        rec["csrt_" + b] = []; rec["gt_" + b] = []

    for i in range(tf, len(frames)):
        frame = cv2.imread(frames[i])
        if frame is None:
            continue
        gt_box = gt[i] if i < len(gt) else None
        ok, bb = tracker.update(frame)
        csrt_box = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])] if ok else None

        rec["frame"].append(i)
        rec["csrt_bbox"].append(csrt_box if csrt_box else [np.nan] * 4)
        rec["gt_bbox"].append(gt_box if gt_box else [np.nan] * 4)
        for src, box in (("csrt", csrt_box), ("gt", gt_box)):
            vecs = None
            if box is not None and box[2] >= args.min_box and box[3] >= args.min_box:
                c = crop(frame, box, pad=0.15)
                if c is not None and c.size:
                    vecs = ext.block_vectors(c)
            for b in BLOCKS:
                rec[src + "_" + b].append(vecs[b] if vecs else None)
        if (i - tf + 1) % 100 == 0:
            print("  %d/%d" % (i - tf + 1, len(frames) - tf))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "uav", args.seq)
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    npz_path = os.path.join(out_dir, "record_%s.npz" % ts)

    save = {"frame": np.array(rec["frame"]),
            "csrt_bbox": np.array(rec["csrt_bbox"], dtype=np.float64),
            "gt_bbox": np.array(rec["gt_bbox"], dtype=np.float64)}
    for b in BLOCKS:
        dim = len(template_vec[b])
        for src in ("csrt", "gt"):
            rows = [v if v is not None else np.full(dim, np.nan)
                    for v in rec[src + "_" + b]]
            save[src + "_" + b] = np.array(rows, dtype=np.float64)
        save["template_" + b] = np.array(template_vec[b], dtype=np.float64)
    np.savez_compressed(npz_path, **save)
    with open(npz_path.replace(".npz", ".json"), "w") as f:
        json.dump({"sequence": args.seq, "n_recorded": len(rec["frame"]),
                   "template": {"frame": tf, "bbox": [int(v) for v in gt[tf]]},
                   "blocks": BLOCKS, "has_gt": True, "min_box": args.min_box}, f, indent=2)
    print("Wrote", npz_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
