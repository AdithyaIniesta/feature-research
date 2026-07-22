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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import ResNet18Blocks, cosine   # noqa: E402,F401
from common.geom import crop                          # noqa: E402
from embedder.embedder_scale import Embedder, load_gt  # noqa: E402

torch.set_num_threads(4)

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
POS_STEPS = 5
SCALES = [0.85, 1.0, 1.18]


def candidates(frame, box):
    """Return (list_of_crops, list_of_boxes) for a local multi-scale search."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    crops, boxes = [], []
    for sf in SCALES:
        nw, nh = w * sf, h * sf
        for dx in np.linspace(-0.4, 0.4, POS_STEPS) * w:
            for dy in np.linspace(-0.4, 0.4, POS_STEPS) * h:
                nb = [cx + dx - nw / 2.0, cy + dy - nh / 2.0, nw, nh]
                c = crop(frame, nb, pad=0.15)
                if c is not None and c.size:
                    crops.append(c)
                    boxes.append([int(round(v)) for v in nb])
    return crops, boxes


def best_box(template_vec, cand_vecs, boxes, fallback):
    """cand_vecs: (N,D) L2-normalizable; pick argmax cosine to template."""
    if not boxes:
        return fallback, -1.0
    t = template_vec / (np.linalg.norm(template_vec) + 1e-8)
    v = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    sims = v @ t
    j = int(np.argmax(sims))
    return boxes[j], float(sims[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--gt-template", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--start", type=int, default=None, help="start/template frame index")
    ap.add_argument("--end", type=int, default=None, help="stop at this frame (exclusive)")
    ap.add_argument("--frame-step", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--fast", action="store_true",
                    help="27 candidates instead of 75 (3x faster, coarser)")
    ap.add_argument("--fuse-weight", type=float, default=0.6,
                    help="fused score = w*embedder + (1-w)*block2 (higher w = trust "
                         "identity more). Blue box.")
    args = ap.parse_args()

    global POS_STEPS, SCALES
    if args.fast:
        POS_STEPS = 3
        SCALES = [0.9, 1.0, 1.1]

    seq_dir = os.path.join(DATA_BASE, args.seq)
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if not frames:
        print("No frames for", args.seq); return 1
    gt = load_gt(args.seq)

    ext = ResNet18Blocks(device="cpu")
    emb = Embedder(args.onnx)

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
    t_res = ext.block2_vectors([tcrop])[0]
    t_emb = emb.embed(tcrop)
    box_res = list(tbox)
    box_emb = list(tbox)
    box_fus = list(tbox)
    w = args.fuse_weight

    def norm_rows(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

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

    paused = False
    i = start
    end = args.end if args.end is not None else len(frames)
    end = min(end, len(frames))
    while i < end:
        frame = cv2.imread(frames[i])
        if frame is None:
            break
        cr_res, bx_res = candidates(frame, box_res)
        if cr_res:
            box_res, s_res = best_box(t_res, ext.block2_vectors(cr_res), bx_res, box_res)
        else:
            s_res = -1.0
        cr_emb, bx_emb = candidates(frame, box_emb)
        if cr_emb:
            box_emb, s_emb = best_box(t_emb, emb.embed_batch(cr_emb), bx_emb, box_emb)
        else:
            s_emb = -1.0

        # fused: same candidates scored by w*embedder + (1-w)*block2 cosine
        cr_fus, bx_fus = candidates(frame, box_fus)
        s_fus = -1.0
        if cr_fus:
            e = norm_rows(emb.embed_batch(cr_fus)) @ (t_emb / (np.linalg.norm(t_emb) + 1e-8))
            r = norm_rows(ext.block2_vectors(cr_fus)) @ (t_res / (np.linalg.norm(t_res) + 1e-8))
            fused = w * e + (1.0 - w) * r
            j = int(np.argmax(fused))
            box_fus, s_fus = bx_fus[j], float(fused[j])

        disp = frame.copy()
        bx, by, bw, bh = box_res
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (0, 230, 0), 2)   # GREEN resnet
        bx, by, bw, bh = box_emb
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (0, 0, 230), 2)   # RED embedder
        bx, by, bw, bh = box_fus
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (255, 150, 0), 2)  # BLUE fused
        cv2.putText(disp, "ResNet block2  cos=%.2f" % s_res, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
        cv2.putText(disp, "Embedder       cos=%.2f" % s_emb, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)
        cv2.putText(disp, "Fused (w=%.1f)  cos=%.2f" % (w, s_fus), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
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
                box_res, box_emb, box_fus = list(tbox), list(tbox), list(tbox)
        except cv2.error:
            pass  # headless: no display, just record

        if not paused:
            i += max(1, args.frame_step)

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
