#!/usr/bin/env python3
"""Side-by-side appearance tracker: GREEN = ResNet18 block2, RED = production embedder.

You select a template once; each frame both methods do a small multi-scale local search
(candidate boxes around their previous position) and pick the box whose crop embedding is
most cosine-similar to the template. Same search for both -- the ONLY difference is the
feature. On a shrinking/receding target you should see RED (embedder) drift or shrink-lock
while GREEN (block2) stays on target, matching the measured scale gap.

Controls (needs a display; on the Jetson run over its monitor):
    c = select template (click-drag)   SPACE = play/pause   r = reset   q = quit

Headless option: --gt-template uses the UAV123 ground-truth box on the start frame as the
template (no clicking) and --record writes an annotated video you can watch later.

Usage (interactive, on Jetson monitor):
  export DISPLAY=:0
  python3 compare/track_compare.py --seq boat6 \
      --onnx /home/nvidia/Music/jetson-tracking-perception/models/embedder_legacy.onnx

Usage (headless video):
  python3 compare/track_compare.py --seq boat6 --gt-template --record \
      --onnx /home/nvidia/Music/jetson-tracking-perception/models/embedder_legacy.onnx
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
from embedder.embedder_scale import Embedder, load_gt  # noqa: E402

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
POS_STEPS = 5
SCALES = [0.85, 1.0, 1.18]


def search(frame, box, template_vec, embed_fn):
    """Local multi-scale search; return (best_box, best_cosine)."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    best, best_s = box, -1.0
    for sf in SCALES:
        nw, nh = w * sf, h * sf
        for dx in np.linspace(-0.4, 0.4, POS_STEPS) * w:
            for dy in np.linspace(-0.4, 0.4, POS_STEPS) * h:
                nb = [cx + dx - nw / 2.0, cy + dy - nh / 2.0, nw, nh]
                c = crop(frame, nb, pad=0.15)
                if c is None or c.size == 0:
                    continue
                s = cosine(template_vec, embed_fn(c))
                if s > best_s:
                    best_s, best = s, nb
    return [int(round(v)) for v in best], best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--gt-template", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--start", type=int, default=None, help="start/template frame index")
    args = ap.parse_args()

    seq_dir = os.path.join(DATA_BASE, args.seq)
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if not frames:
        print("No frames for", args.seq); return 1
    gt = load_gt(args.seq)

    ext = ResNet18Blocks(device="cpu")
    emb = Embedder(args.onnx)
    resnet_fn = ext.block2_vector
    embed_fn = emb.embed

    start = args.start if args.start is not None else \
        (next((i for i, b in enumerate(gt) if b), 0) if (args.gt_template and gt) else 0)

    # --- get template box ---
    tframe = cv2.imread(frames[start])
    if args.gt_template and gt and gt[start]:
        tbox = [int(v) for v in gt[start]]
    else:
        cv2.namedWindow("track", cv2.WINDOW_AUTOSIZE)
        print("Select template: click-drag, ENTER; then it tracks.")
        roi = cv2.selectROI("track", tframe, False, False)
        if not roi or roi[2] < 4:
            print("No template selected."); return 1
        tbox = [int(v) for v in roi]

    tcrop = crop(tframe, tbox, pad=0.15)
    t_res = resnet_fn(tcrop)
    t_emb = embed_fn(tcrop)
    box_res = list(tbox)
    box_emb = list(tbox)

    writer = None
    out_path = None
    if args.record:
        from viz.vizutil import open_writer
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "compare")
        os.makedirs(out_dir, exist_ok=True)
        h0, w0 = tframe.shape[:2]
        writer, out_path = open_writer(os.path.join(out_dir, "track_%s.mp4" % args.seq),
                                       12, w0, h0)

    show = not args.record or True  # still show if display available
    paused = False
    i = start
    while i < len(frames):
        frame = cv2.imread(frames[i])
        if frame is None:
            break
        box_res, s_res = search(frame, box_res, t_res, resnet_fn)
        box_emb, s_emb = search(frame, box_emb, t_emb, embed_fn)

        disp = frame.copy()
        x, y, w, h = box_res
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 230, 0), 2)      # GREEN resnet
        x, y, w, h = box_emb
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 230), 2)      # RED embedder
        cv2.putText(disp, "ResNet block2  cos=%.2f" % s_res, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
        cv2.putText(disp, "Embedder       cos=%.2f" % s_emb, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)
        cv2.putText(disp, "frame %d" % i, (10, disp.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer is not None:
            writer.write(disp)
        try:
            cv2.imshow("track", disp)
            key = cv2.waitKey(0 if paused else 20) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            if key in (ord('r'), ord('R')):
                box_res, box_emb = list(tbox), list(tbox)
        except cv2.error:
            pass  # headless: no display, just record

        if not paused:
            i += 1

    if writer is not None:
        writer.release()
        print("Wrote", out_path)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
