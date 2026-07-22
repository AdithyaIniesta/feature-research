"""Experiment 7: Automatic handoff analysis.

For each HANDOFF event (stereo camera switch), quantify around the handoff:
  * feature_similarity : cosine sim between the target template on the SOURCE camera
                         (just before handoff) and the target crop on the DESTINATION
                         camera (at handoff), per ResNet18 block.
  * post_iou           : IoU of the destination tracker box vs manual annotation,
                         averaged over a short window after handoff.
  * lock_delay_frames  : frames from handoff until the tracker reports TRACKING again
                         (via MODE_CHANGE), if observable.

Goal: does feature similarity predict a successful handoff (high post_iou, short lock)?

Output:
  results/exp07/handoff_metrics.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataio import Session          # noqa: E402
from common.features import ResNet18Blocks, cosine  # noqa: E402
from common.geom import crop, iou, xywh_to_xyxy      # noqa: E402

BLOCKS = ["block1", "block2", "block3", "block4"]

# camera-id -> side.  From meta/events: cam 1 = left, cam 2 = right (direction R2L
# means from_cam=2/right -> to_cam=1/left).
CAM_SIDE = {1: "left", 2: "right"}


def nearest_tracker_frame(sess, cam, target_frame, search=30):
    """Find the closest frame <= target_frame (then >) that has a tracker box for cam."""
    for delta in range(0, search + 1):
        for fr in (target_frame - delta, target_frame + delta):
            if sess.tracker_box(cam, fr) is not None:
                try:
                    sess.read_frame(cam, fr)
                except FileNotFoundError:
                    continue
                return fr
    return None


def post_lock_delay(sess, dst_cam_id, handoff_frame):
    """Frames until a MODE_CHANGE to TRACKING on dst cam after handoff. None if not found."""
    best = None
    for e in sess.mode_changes():
        if e.get("to") == "TRACKING" and e.get("cam") == dst_cam_id:
            fr = e.get("frame", -1)
            if fr >= handoff_frame:
                d = fr - handoff_frame
                if best is None or d < best:
                    best = d
    return best


def avg_post_iou(sess, cam, start_frame, window=20):
    ious = []
    for fr in range(start_frame, start_frame + window):
        tb = sess.tracker_box(cam, fr)
        ab = sess.annotation_box(cam, fr)
        if tb is None or ab is None:
            continue
        ious.append(iou(xywh_to_xyxy(tb), ab))
    return float(np.mean(ious)) if ious else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp07")
    os.makedirs(out_dir, exist_ok=True)

    sess = Session(args.session)
    ext = ResNet18Blocks(device="cpu")

    handoffs = sess.handoffs()
    if not handoffs:
        print("No HANDOFF events in session.")
        return
    print("Found %d handoff(s)." % len(handoffs))

    rows = []
    for h in handoffs:
        hf = h.get("frame")
        src_id = h.get("from_cam")
        dst_id = h.get("to_cam")
        src_cam = CAM_SIDE.get(src_id)
        dst_cam = CAM_SIDE.get(dst_id)
        direction = h.get("direction", "")

        # --- source template (tracker box on source cam, just before handoff) ---
        src_fr = nearest_tracker_frame(sess, src_cam, hf)
        # --- destination crop: box given directly in the HANDOFF event ----------
        dst_box = [h.get("dst_x"), h.get("dst_y"), h.get("dst_w"), h.get("dst_h")]
        # dst_x/dst_y are center-ish target coords; treat as top-left of given w/h.
        # (Adjust here if the recorder emits center coords — flagged for the user.)

        sims = {b: float("nan") for b in BLOCKS}
        if src_fr is not None and None not in dst_box:
            src_img = sess.read_frame(src_cam, src_fr)
            src_crop = crop(src_img, sess.tracker_box(src_cam, src_fr), pad=0.15)
            try:
                dst_img = sess.read_frame(dst_cam, hf)
            except FileNotFoundError:
                dst_img = None
            dst_crop = crop(dst_img, dst_box, pad=0.15) if dst_img is not None else None
            if src_crop is not None and dst_crop is not None \
                    and src_crop.size and dst_crop.size:
                sv = ext.block_vectors(src_crop)
                dv = ext.block_vectors(dst_crop)
                sims = {b: cosine(sv[b], dv[b]) for b in BLOCKS}

        post_iou = avg_post_iou(sess, dst_cam, hf, args.window)
        lock = post_lock_delay(sess, dst_id, hf)

        row = {
            "handoff_frame": hf,
            "direction": direction,
            "src_cam": src_cam,
            "dst_cam": dst_cam,
            "src_template_frame": src_fr,
            "post_iou": post_iou,
            "lock_delay_frames": lock,
        }
        row.update({("sim_" + b): sims[b] for b in BLOCKS})
        rows.append(row)
        print("  handoff@%s %s: sim_block4=%.3f post_iou=%s lock=%s"
              % (hf, direction, sims["block4"],
                 "%.3f" % post_iou if post_iou is not None else "NA",
                 lock if lock is not None else "NA"))

    # --- write CSV -------------------------------------------------------
    cols = ["handoff_frame", "direction", "src_cam", "dst_cam",
            "src_template_frame", "post_iou", "lock_delay_frames"] \
        + ["sim_" + b for b in BLOCKS]
    csv_path = os.path.join(out_dir, "handoff_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("Wrote", csv_path)
    print("\nNOTE: dst_x/dst_y are treated as the top-left of the handoff box. If the "
          "recorder emits CENTER coords, tell me and I'll shift by w/2,h/2.")


if __name__ == "__main__":
    main()
