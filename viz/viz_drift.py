"""Replay video: watch feature-similarity vs pixel-similarity drift over a session.

Plays the left-camera frames, draws the tracker box, and shows two live meters:
  * FEATURE  = ResNet18 block2 cosine similarity of the current target vs the FIRST target
  * PIXEL    = classic normalized-cross-correlation of the same crops (old RGB template)

As the target changes size/appearance you should SEE the pixel meter drop faster than the
feature meter -- the whole point of the research, made visual.

Output: results/viz/drift_left.mp4  (or .avi fallback)
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
from viz.vizutil import ncc, draw_meter, draw_box, paste_thumb, open_writer  # noqa: E402

BLOCK = "block2"  # good scale-invariance vs discriminability trade-off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cam", default="left", choices=["left", "right"])
    ap.add_argument("--step", type=int, default=3, help="process every Nth target (speed)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "viz")
    os.makedirs(out_dir, exist_ok=True)

    sess = Session(args.session)
    ext = ResNet18Blocks(device="cpu")

    angles = sess.angle_events(args.cam)
    angles = [a for a in angles if a["box"][2] >= 12 and a["box"][3] >= 12]
    angles = angles[::args.step]
    if not angles:
        print("No usable targets for cam=%s" % args.cam)
        return
    print("Rendering %d frames..." % len(angles))

    # reference = first target crop
    ref_crop = None
    ref_vec = None
    writer = None
    out_path = os.path.join(out_dir, "drift_%s.mp4" % args.cam)

    for i, a in enumerate(angles):
        idx = sess.index_for_time(args.cam, a["t_ns"])
        try:
            frame = sess.read_frame(args.cam, idx)
        except FileNotFoundError:
            continue
        c = crop(frame, a["box"], pad=0.15)
        if c is None or c.size == 0:
            continue

        if ref_crop is None:
            ref_crop = c.copy()
            ref_vec = ext.block_vectors(ref_crop)

        feat = cosine(ref_vec[BLOCK], ext.block_vectors(c)[BLOCK])
        pix = ncc(ref_crop, c)

        canvas = frame.copy()
        draw_box(canvas, a["box"], (0, 255, 0), "target")
        # panel background
        cv2.rectangle(canvas, (10, 10), (430, 150), (0, 0, 0), -1)
        cv2.addWeighted(canvas, 1.0, canvas, 0.0, 0)
        draw_meter(canvas, 20, 45, 300, 22, feat, "FEATURE(block2)", (0, 220, 0))
        draw_meter(canvas, 20, 105, 300, 22, pix, "PIXEL(NCC)", (0, 140, 255))
        paste_thumb(canvas, ref_crop, 340, 40, box=80, label="ref")
        cv2.putText(canvas, "frame %d  size %dx%d" % (idx, a["box"][2], a["box"][3]),
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            h, w = canvas.shape[:2]
            writer, out_path = open_writer(out_path, args.fps, w, h)
        writer.write(canvas)
        if (i + 1) % 30 == 0:
            print("  %d/%d  feat=%.2f pix=%.2f" % (i + 1, len(angles), feat, pix))

    if writer is not None:
        writer.release()
        print("Wrote", out_path)
    else:
        print("No frames written.")


if __name__ == "__main__":
    main()
