"""Replay clips around each stereo HANDOFF.

For every HANDOFF event, builds a short clip of the DESTINATION camera frames right after
the switch, with the tracker box drawn, and overlays how similar each frame's target is to
the SOURCE template (the target as it looked just before the hop). The source template
thumbnail sits in the corner so you can eyeball "is this still the same object?".

Output: results/viz/handoff_<frame>_<dir>.mp4  (one per handoff)
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataio import Session, CAM_SIDE  # noqa: E402
from common.features import ResNet18Blocks, cosine  # noqa: E402
from common.geom import crop                 # noqa: E402
from viz.vizutil import ncc, draw_meter, draw_box, paste_thumb, open_writer  # noqa: E402

BLOCK = "block2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--pre-ns", type=float, default=1.0e9, help="source window before (ns)")
    ap.add_argument("--post-ns", type=float, default=2.0e9, help="dest window after (ns)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "viz")
    os.makedirs(out_dir, exist_ok=True)

    sess = Session(args.session)
    ext = ResNet18Blocks(device="cpu")

    handoffs = sess.handoffs()
    if not handoffs:
        print("No HANDOFF events.")
        return

    for h in handoffs:
        h_t = h.get("t_ns", 0)
        src_cam = CAM_SIDE.get(h.get("from_cam"))
        dst_cam = CAM_SIDE.get(h.get("to_cam"))
        direction = h.get("direction", "")

        # source template = nearest ANGLE on source side before the hop
        src_a = sess.nearest_angle(src_cam, h_t, direction="before")
        if src_a is None:
            print("handoff@%s: no source template, skipping" % h.get("frame"))
            continue
        s_idx = sess.index_for_time(src_cam, src_a["t_ns"])
        src_crop = crop(sess.read_frame(src_cam, s_idx), src_a["box"], pad=0.15)
        src_vec = ext.block_vectors(src_crop)

        # destination frames within the post window
        dst_angles = [a for a in sess.angle_events(dst_cam)
                      if 0 <= (a["t_ns"] - h_t) <= args.post_ns]
        if not dst_angles:
            print("handoff@%s: no destination frames, skipping" % h.get("frame"))
            continue

        out_path = os.path.join(out_dir, "handoff_%s_%s.mp4" % (h.get("frame"), direction))
        writer = None
        for a in dst_angles:
            d_idx = sess.index_for_time(dst_cam, a["t_ns"])
            try:
                frame = sess.read_frame(dst_cam, d_idx)
            except FileNotFoundError:
                continue
            c = crop(frame, a["box"], pad=0.15)
            feat = pix = 0.0
            if c is not None and c.size:
                feat = cosine(src_vec[BLOCK], ext.block_vectors(c)[BLOCK])
                pix = ncc(src_crop, c)

            canvas = frame.copy()
            draw_box(canvas, a["box"], (0, 255, 255), "dst target")
            cv2.rectangle(canvas, (10, 10), (430, 150), (0, 0, 0), -1)
            draw_meter(canvas, 20, 45, 300, 22, feat, "FEATURE vs src", (0, 220, 0))
            draw_meter(canvas, 20, 105, 300, 22, pix, "PIXEL vs src", (0, 140, 255))
            paste_thumb(canvas, src_crop, 340, 40, box=80, label="src")
            cv2.putText(canvas, "HANDOFF %s  dst frame %d" % (direction, d_idx),
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if writer is None:
                hh, ww = canvas.shape[:2]
                writer, out_path = open_writer(out_path, args.fps, ww, hh)
            writer.write(canvas)

        if writer is not None:
            writer.release()
            print("Wrote", out_path)


if __name__ == "__main__":
    main()
