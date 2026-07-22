#!/usr/bin/env python3
"""Interactive UAV123 recorder (based on ltmu-jetson's ltmu_uav_demo.py).

Flow: pick a UAV123 sequence -> play -> press 'c' and click-drag the template you want
-> a CSRT tracker follows it. From that point every frame we RECORD:
  * csrt_bbox        : the box CSRT reports (real tracker drift)
  * gt_bbox          : the UAV123 ground-truth box for that frame (perfect box)
  * block1..4 features of BOTH crops (ResNet18), plus the template's features
So one recording lets you later compare layer stability under real drift vs the true box.

Controls:  SPACE play/pause   c: select template   r: reset   q: quit
Output:    results/uav/<seq>/record_<ts>.npz  (+ .json meta)

Requires a display (run on the Jetson monitor). CPU feature extraction makes playback
slower than 30 fps -- that's expected.
"""
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


def select_sequence():
    if not os.path.isdir(DATA_BASE):
        print("ERROR: %s not found" % DATA_BASE)
        return None, None
    seqs = [d for d in sorted(os.listdir(DATA_BASE))
            if os.path.isdir(os.path.join(DATA_BASE, d))]
    print("\nAvailable sequences:")
    for i, s in enumerate(seqs, 1):
        print("  [%d] %s" % (i, s))
    while True:
        try:
            c = int(input("\nSelect [1-%d]: " % len(seqs)))
            if 1 <= c <= len(seqs):
                return os.path.join(DATA_BASE, seqs[c - 1]), seqs[c - 1]
        except Exception:
            pass
        print("Invalid choice.")


def load_gt(seq_name, n_frames):
    """Load UAV123 ground-truth boxes (x,y,w,h per line). Returns list len n_frames or None."""
    p = os.path.join(ANNO_BASE, seq_name + ".txt")
    if not os.path.isfile(p):
        print("[GT] no annotation file %s -- GT disabled" % p)
        return None
    boxes = []
    with open(p) as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                boxes.append(None); continue
            try:
                x, y, w, h = [float(v) for v in line.split()[:4]]
                if any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0:
                    boxes.append(None)
                else:
                    boxes.append([x, y, w, h])
            except Exception:
                boxes.append(None)
    if len(boxes) != n_frames:
        print("[GT] warning: %d gt lines vs %d frames; aligning from start"
              % (len(boxes), n_frames))
    return boxes


def make_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


def main():
    if len(sys.argv) > 1:
        seq_dir = sys.argv[1]
        seq_name = os.path.basename(seq_dir.rstrip("/"))
    else:
        seq_dir, seq_name = select_sequence()
    if not seq_dir:
        return 1

    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if not frames:
        print("ERROR: no images in %s" % seq_dir)
        return 1
    print("[SEQ] %s: %d frames" % (seq_name, len(frames)))
    gt = load_gt(seq_name, len(frames))

    ext = ResNet18Blocks(device="cpu")

    rec = {"frame": [], "csrt_bbox": [], "gt_bbox": []}
    for b in BLOCKS:
        rec["csrt_" + b] = []
        rec["gt_" + b] = []
    template_vec = None
    template_meta = {}

    tracker = None
    win = "UAV123 recorder"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("[MAIN] SPACE play/pause   c: select template   r: reset   q: quit")

    idx, paused = 0, False
    while True:
        frame = cv2.imread(frames[idx])
        if frame is None:
            break
        disp = frame.copy()

        gt_box = gt[idx] if (gt and idx < len(gt)) else None
        if gt_box:
            x, y, w, h = [int(v) for v in gt_box]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 180, 0), 1)

        csrt_box = None
        if tracker is not None:
            ok, bb = tracker.update(frame)
            if ok:
                csrt_box = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
                x, y, w, h = csrt_box
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.putText(disp, "TRACKING", (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
            else:
                cv2.putText(disp, "LOST", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # --- record this frame (features for csrt + gt crops) ---
            rec["frame"].append(idx)
            rec["csrt_bbox"].append(csrt_box if csrt_box else [np.nan] * 4)
            rec["gt_bbox"].append(gt_box if gt_box else [np.nan] * 4)
            for src, box in (("csrt", csrt_box), ("gt", gt_box)):
                vecs = None
                if box is not None:
                    c = crop(frame, box, pad=0.15)
                    if c is not None and c.size:
                        vecs = ext.block_vectors(c)
                for b in BLOCKS:
                    rec[src + "_" + b].append(vecs[b] if vecs else None)

        cv2.putText(disp, "%s | f%d/%d  SPACE c r q"
                    % ("PAUSE" if paused else "PLAY", idx, len(frames) - 1),
                    (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(0 if paused else 25) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key in (ord('c'), ord('C')):
            roi = cv2.selectROI(win, frame, False, False)
            cv2.destroyWindow("ROI selector")
            if roi and roi[2] > 4 and roi[3] > 4:
                tracker = make_csrt()
                tracker.init(frame, tuple(int(v) for v in roi))
                tc = crop(frame, list(roi), pad=0.15)
                template_vec = ext.block_vectors(tc)
                template_meta = {"frame": idx, "bbox": [int(v) for v in roi]}
                print("[MAIN] template @ frame %d bbox %s" % (idx, list(roi)))
        elif key in (ord('r'), ord('R')):
            tracker = None
            print("[MAIN] reset")
        elif key in (ord('q'), ord('Q'), 27):
            break

        if not paused:
            idx += 1
            if idx >= len(frames):
                break

    cv2.destroyAllWindows()

    if not rec["frame"]:
        print("Nothing recorded (no template selected).")
        return 0

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "uav", seq_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    npz_path = os.path.join(out_dir, "record_%s.npz" % ts)

    # pad missing feature vectors with NaN rows so arrays are rectangular
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
        json.dump({"sequence": seq_name, "n_recorded": len(rec["frame"]),
                   "template": template_meta, "blocks": BLOCKS,
                   "has_gt": gt is not None}, f, indent=2)
    print("Wrote", npz_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
