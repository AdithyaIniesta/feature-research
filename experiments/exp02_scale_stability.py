"""Experiment 2: Feature stability vs scale.

Target crops are taken from ANGLE events (dense per-frame tracker boxes), mapped to the
correct jpg by timestamp. Each crop is rescaled to several factors; ResNet18 block1..4
features are compared (cosine) against the 1.0x reference.

Output: which block holds up best as the target changes size.
  results/exp02/scale_stability.csv
  results/exp02/scale_stability.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataio import Session          # noqa: E402
from common.features import ResNet18Blocks, cosine  # noqa: E402
from common.geom import crop               # noqa: E402

SCALES = [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0]
BLOCKS = ["block1", "block2", "block3", "block4"]


def rescale(img, s):
    if s == 1.0:
        return img
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def sample_targets(sess, side, max_targets, min_size=16):
    angles = [a for a in sess.angle_events(side)
              if a["box"][2] >= min_size and a["box"][3] >= min_size]
    if not angles:
        return []
    if len(angles) > max_targets:
        idx = np.linspace(0, len(angles) - 1, max_targets).astype(int)
        angles = [angles[i] for i in idx]
    return angles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cam", default="left", choices=["left", "right"])
    ap.add_argument("--max-targets", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp02")
    os.makedirs(out_dir, exist_ok=True)

    sess = Session(args.session)
    ext = ResNet18Blocks(device="cpu")

    targets = sample_targets(sess, args.cam, args.max_targets)
    if not targets:
        print("No usable ANGLE targets found for cam=%s" % args.cam)
        return
    print("Using %d target crops (cam=%s, from ANGLE events)." % (len(targets), args.cam))

    sims = {b: {s: [] for s in SCALES} for b in BLOCKS}
    used = 0
    for ti, a in enumerate(targets):
        idx = sess.index_for_time(args.cam, a["t_ns"])
        try:
            img = sess.read_frame(args.cam, idx)
        except FileNotFoundError:
            continue
        base = crop(img, a["box"], pad=0.15)
        if base is None or base.shape[0] < 8 or base.shape[1] < 8:
            continue
        ref = ext.block_vectors(base)
        for s in SCALES:
            vec = ext.block_vectors(rescale(base, s))
            for b in BLOCKS:
                sims[b][s].append(cosine(ref[b], vec[b]))
        used += 1
        if used % 15 == 0:
            print("  processed %d/%d" % (used, len(targets)))

    print("Crops actually used: %d" % used)

    csv_path = os.path.join(out_dir, "scale_stability.csv")
    with open(csv_path, "w") as f:
        f.write("block,scale,mean_cosine,std_cosine,n\n")
        for b in BLOCKS:
            for s in SCALES:
                arr = np.array(sims[b][s], dtype=np.float64)
                if arr.size == 0:
                    continue
                f.write("%s,%.3f,%.4f,%.4f,%d\n"
                        % (b, s, arr.mean(), arr.std(), arr.size))
    print("Wrote", csv_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for b in BLOCKS:
            means = [np.mean(sims[b][s]) if sims[b][s] else np.nan for s in SCALES]
            plt.plot(SCALES, means, marker="o", label=b)
        plt.axvline(1.0, color="gray", ls="--", lw=1)
        plt.xlabel("scale factor")
        plt.ylabel("mean cosine similarity vs 1.0x")
        plt.title("Feature stability vs scale (ResNet18 blocks, n=%d)" % used)
        plt.legend()
        plt.grid(True, alpha=0.3)
        png = os.path.join(out_dir, "scale_stability.png")
        plt.savefig(png, dpi=130, bbox_inches="tight")
        print("Wrote", png)
    except Exception as e:
        print("Plot skipped:", e)

    print("\nMost scale-invariant block (higher = better, avg over non-1.0 scales):")
    ranking = []
    for b in BLOCKS:
        vals = [np.mean(sims[b][s]) for s in SCALES if s != 1.0 and sims[b][s]]
        ranking.append((b, float(np.mean(vals)) if vals else 0.0))
    for b, v in sorted(ranking, key=lambda x: -x[1]):
        print("  %-7s %.4f" % (b, v))


if __name__ == "__main__":
    main()
