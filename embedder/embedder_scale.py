#!/usr/bin/env python3
"""Repeat the scale-invariance test on the PRODUCTION tracker's embedder
(embedder_legacy.onnx) and compare it to ResNet18 block2.

The tracker's verifier (cvtracker FeatureExtractor / TrtFeatureExtractor) crops the
tracked object, resizes it to 128x128 GRAYSCALE (replicated to 3 channels), ImageNet-
normalizes, embeds, L2-normalizes, and compares templates by COSINE similarity for
drift/distractor rejection and LOST re-capture. So the operational question is: does that
embedding stay stable as the same object appears at different sizes?

Because the crop is always resized to 128x128, input-size scale is normalized away; the
residual effect is RESOLUTION. We simulate the object captured at linear scale s by
resizing the source crop to s x its size (s<1 = far/blurry, s>1 = near) and then feeding
it through the fixed 128 pipeline. We report cosine vs the s=1 reference, averaged over
many target crops, for the embedder AND for ResNet18 block2 on the identical crops.

Usage:
  python3 embedder/embedder_scale.py \
      --onnx /home/nvidia/Music/jetson-tracking-perception/models/embedder_legacy.onnx \
      --seqs boat6 uav5 car1 person1 --per-seq 15
Output: results/embedder/embedder_scale.csv + .png
Requires: onnxruntime  (pip3 install onnxruntime  if missing)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import ResNet18Blocks, cosine   # noqa: E402
from common.geom import crop                          # noqa: E402

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 128
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


class Embedder:
    """embedder_legacy.onnx via onnxruntime, matching the tracker's preprocessing."""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4      # avoids the pthread_setaffinity warning on Jetson
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        # is the batch dimension dynamic? (else we must run one crop at a time)
        shp = self.sess.get_inputs()[0].shape
        self.dynamic_batch = not isinstance(shp[0], int) or shp[0] < 1

    def _blob(self, crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32)  # 3,128,128

    def embed(self, crop_bgr):
        out = self.sess.run([self.out_name], {self.in_name: self._blob(crop_bgr)[None]})[0]
        v = out.flatten().astype(np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_batch(self, crops_bgr):
        """Return L2-normalized embeddings (N, D). Batches if the model allows it."""
        blobs = np.stack([self._blob(c) for c in crops_bgr])  # (N,3,128,128)
        if self.dynamic_batch:
            out = self.sess.run([self.out_name], {self.in_name: blobs})[0]
        else:
            out = np.stack([self.sess.run([self.out_name],
                                          {self.in_name: b[None]})[0].flatten()
                            for b in blobs])
        out = out.reshape(len(crops_bgr), -1).astype(np.float64)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def resample(cimg, s):
    if s == 1.0:
        return cimg
    h, w = cimg.shape[:2]
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(cimg, (nw, nh), interpolation=interp)


def sample_crops(seqs, per_seq, min_size):
    crops = []
    for seq in seqs:
        seq_dir = os.path.join(DATA_BASE, seq)
        frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
            sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        gt = load_gt(seq)
        if not frames or gt is None:
            print("skip", seq); continue
        idx = [i for i, b in enumerate(gt) if b and min(b[2], b[3]) >= min_size]
        if not idx:
            continue
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, min(per_seq, len(idx))).astype(int)]
        for i in idx:
            img = cv2.imread(frames[i])
            if img is None:
                continue
            c = crop(img, gt[i], pad=0.15)
            if c is not None and c.shape[0] >= min_size and c.shape[1] >= min_size:
                crops.append(c)
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--per-seq", type=int, default=15)
    ap.add_argument("--min-size", type=int, default=40)
    ap.add_argument("--no-resnet", action="store_true")
    args = ap.parse_args()

    emb = Embedder(args.onnx)
    ext = None if args.no_resnet else ResNet18Blocks(device="cpu")

    crops = sample_crops(args.seqs, args.per_seq, args.min_size)
    if not crops:
        print("No crops."); return 1
    print("Using %d target crops from %s" % (len(crops), ",".join(args.seqs)))

    emb_sims = {s: [] for s in SCALES}
    res_sims = {s: [] for s in SCALES}
    for ci, c in enumerate(crops):
        e_ref = emb.embed(c)
        r_ref = ext.block_vectors(c)["block2"] if ext else None
        for s in SCALES:
            cs = resample(c, s)
            emb_sims[s].append(cosine(e_ref, emb.embed(cs)))
            if ext:
                res_sims[s].append(cosine(r_ref, ext.block_vectors(cs)["block2"]))
        if (ci + 1) % 25 == 0:
            print("  %d/%d" % (ci + 1, len(crops)))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "embedder")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "embedder_scale.csv")
    with open(csv_path, "w") as f:
        f.write("scale,embedder_mean,embedder_std,resnet_block2_mean\n")
        for s in SCALES:
            em = np.mean(emb_sims[s]); es = np.std(emb_sims[s])
            rm = np.mean(res_sims[s]) if ext else float("nan")
            f.write("%.2f,%.4f,%.4f,%.4f\n" % (s, em, es, rm))
    print("Wrote", csv_path)

    print("\nScale robustness (cosine vs 1.0x; higher = more scale-invariant):")
    print("  %6s %14s %16s" % ("scale", "EMBEDDER", "ResNet block2"))
    for s in SCALES:
        rm = "%.3f" % np.mean(res_sims[s]) if ext else "  n/a"
        print("  %6.2f %14.3f %16s" % (s, np.mean(emb_sims[s]), rm))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.errorbar(SCALES, [np.mean(emb_sims[s]) for s in SCALES],
                     yerr=[np.std(emb_sims[s]) for s in SCALES],
                     marker="o", capsize=3, label="production embedder")
        if ext:
            plt.plot(SCALES, [np.mean(res_sims[s]) for s in SCALES],
                     marker="s", label="ResNet18 block2")
        plt.axvline(1.0, color="gray", ls="--", lw=1)
        plt.xscale("log")
        plt.xlabel("object capture scale vs reference (log)")
        plt.ylabel("cosine similarity vs 1.0x")
        plt.title("Embedding scale robustness: production embedder vs ResNet18 block2")
        plt.legend(); plt.grid(True, alpha=0.3, which="both")
        png = os.path.join(out_dir, "embedder_scale.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print("Wrote", png)
    except Exception as e:
        print("Plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
