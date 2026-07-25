#!/usr/bin/env python3
"""Go/no-go test for the DINOv2 scale-pick box-size idea.

The scale-pick works ONLY if DINOv2's cosine similarity has a sharp enough peak at the
correct box size. DINOv2 is deliberately scale-tolerant (good verifier), which could
flatten that peak and make size selection unreliable — so measure it before writing any
C++.

Method: for each ground-truth target box, take the GT-size crop as the "template", then
crop the SAME target center at several box scales (0.7x..1.3x — bigger box = more
background, smaller = target cropped), embed each with DINOv2, and measure cosine vs the
template. Average over many targets.

Read the printed curve:
  * SHARP peak at 1.0x (cosine drops clearly by 0.8x/1.2x) -> scale-pick is viable, build it.
  * FLAT (cosine stays ~0.98 across scales)              -> too flat, use a scale pyramid
                                                            or a trained head instead.

Uses the SAME preprocessing the production verifier uses (gray -> 3ch, 224, ImageNet norm)
so the result reflects the real pipeline.

Usage (on the Jetson):
  python3 uav/dino_scale_pick_test.py \
      --onnx /home/nvidia/Music/jetson-tracking-perception/models/dinov2_small.onnx \
      --seqs boat6 uav5 car16 person1 --per-seq 25
Output: results/dino_scale_pick/curve.csv + verdict printed
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.geom import crop           # noqa: E402
from common.features import cosine     # noqa: E402

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 224
SCALES = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]


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


class Dino:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, crop_bgr):
        # Match production verifier: grayscale replicated to 3ch, 224, ImageNet norm.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        out = self.sess.run([self.out_name], {self.in_name: blob})[0].flatten()
        n = np.linalg.norm(out)
        return out.astype(np.float64) / n if n > 0 else out.astype(np.float64)


def scaled_box(b, s):
    x, y, w, h = b
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * s, h * s
    return [cx - nw / 2.0, cy - nh / 2.0, nw, nh]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--per-seq", type=int, default=25)
    ap.add_argument("--min-size", type=int, default=30)
    args = ap.parse_args()

    dino = Dino(args.onnx)
    sims = {s: [] for s in SCALES}
    total = 0

    for seq in args.seqs:
        seq_dir = os.path.join(DATA_BASE, seq)
        frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
            sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        gt = load_gt(seq)
        if not frames or gt is None:
            print("skip", seq); continue
        idx = [i for i, b in enumerate(gt) if b and min(b[2], b[3]) >= args.min_size]
        if not idx:
            continue
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1,
                                           min(args.per_seq, len(idx))).astype(int)]
        print("[%s] %d targets" % (seq, len(idx)))
        for i in idx:
            img = cv2.imread(frames[i])
            if img is None:
                continue
            tmpl_crop = crop(img, gt[i], pad=0.0)           # GT-size template
            if tmpl_crop is None or tmpl_crop.size == 0:
                continue
            tmpl = dino.embed(tmpl_crop)
            for s in SCALES:
                c = crop(img, scaled_box(gt[i], s), pad=0.0)
                if c is None or c.size == 0:
                    continue
                sims[s].append(cosine(tmpl, dino.embed(c)))
            total += 1

    if total == 0:
        print("No targets."); return 1

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "dino_scale_pick")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "curve.csv"), "w") as f:
        f.write("scale,mean_cosine,std\n")
        for s in SCALES:
            a = np.array(sims[s])
            f.write("%.2f,%.4f,%.4f\n" % (s, a.mean(), a.std()))

    print("\nDINOv2 cosine vs box scale (n=%d):" % total)
    means = {}
    for s in SCALES:
        m = float(np.mean(sims[s]))
        means[s] = m
        bar = "#" * int(round((m - 0.7) / 0.3 * 40)) if m > 0.7 else ""
        print("  %.2fx  %.4f  %s" % (s, m, bar))

    # Verdict: how much does cosine drop from 1.0x to the +-20% neighbours?
    drop = means[1.0] - 0.5 * (means[0.8] + means[1.2])
    print("\n  peak drop (1.0x vs mean of 0.8x/1.2x) = %.4f" % drop)
    if drop >= 0.03:
        print("  VERDICT: peak is sharp enough → scale-pick is VIABLE. Build it in C++.")
    elif drop >= 0.012:
        print("  VERDICT: marginal → scale-pick may work but coarse; test a pyramid too.")
    else:
        print("  VERDICT: peak too FLAT → scale-pick unreliable. Use a scale pyramid or a "
              "trained head instead.")
    print("\n(Note: this used GRAYSCALE input like production. If flat, rerun idea with "
          "RGB — DINOv2 was RGB-trained and colour may sharpen the size signal.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
