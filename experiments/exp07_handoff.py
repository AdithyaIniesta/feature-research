"""Experiment 7: Automatic handoff analysis.

For each HANDOFF event (stereo camera switch), quantify around the handoff:
  * feature_similarity : cosine sim (per ResNet18 block) between the target template on
                         the SOURCE camera just BEFORE handoff and the target crop on the
                         DESTINATION camera just AFTER handoff. Both crops come from ANGLE
                         events (dense boxes), mapped to jpgs by timestamp.
  * src_det / dst_det  : tracker confidence (ANGLE `det`) on each side around handoff.
  * lock_delay_frames  : frames from handoff until MODE_CHANGE back to TRACKING on dst cam.

Goal: does feature similarity predict a successful handoff (target re-acquired, high
confidence, short lock)?

NOTE: annotations.json in this session has only ~5 frames, too sparse for per-handoff
IoU, so IoU is omitted here (covered later once denser ground truth exists).

Output: results/exp07/handoff_metrics.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataio import Session, CAM_SIDE  # noqa: E402
from common.features import ResNet18Blocks, cosine  # noqa: E402
from common.geom import crop                 # noqa: E402

BLOCKS = ["block1", "block2", "block3", "block4"]


def crop_from_angle(sess, ext, side, angle):
    if angle is None:
        return None
    idx = sess.index_for_time(side, angle["t_ns"])
    try:
        img = sess.read_frame(side, idx)
    except FileNotFoundError:
        return None
    c = crop(img, angle["box"], pad=0.15)
    if c is None or c.size == 0:
        return None
    return ext.block_vectors(c)


def post_lock_delay(sess, dst_cam_id, handoff_frame):
    best = None
    for e in sess.mode_changes():
        if e.get("to") == "TRACKING" and e.get("cam") == dst_cam_id:
            fr = e.get("frame", -1)
            if fr >= handoff_frame:
                d = fr - handoff_frame
                if best is None or d < best:
                    best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
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
        h_t = h.get("t_ns", 0)
        src_id, dst_id = h.get("from_cam"), h.get("to_cam")
        src_cam, dst_cam = CAM_SIDE.get(src_id), CAM_SIDE.get(dst_id)
        direction = h.get("direction", "")

        src_angle = sess.nearest_angle(src_cam, h_t, direction="before")
        dst_angle = sess.nearest_angle(dst_cam, h_t, direction="after")

        sv = crop_from_angle(sess, ext, src_cam, src_angle)
        dv = crop_from_angle(sess, ext, dst_cam, dst_angle)

        sims = {b: float("nan") for b in BLOCKS}
        if sv is not None and dv is not None:
            sims = {b: cosine(sv[b], dv[b]) for b in BLOCKS}

        lock = post_lock_delay(sess, dst_id, hf)
        row = {
            "handoff_frame": hf,
            "direction": direction,
            "src_cam": src_cam,
            "dst_cam": dst_cam,
            "src_det": src_angle["det"] if src_angle else None,
            "dst_det": dst_angle["det"] if dst_angle else None,
            "lock_delay_frames": lock,
        }
        row.update({("sim_" + b): sims[b] for b in BLOCKS})
        rows.append(row)
        print("  handoff@%s %s: sim_b4=%.3f src_det=%s dst_det=%s lock=%s"
              % (hf, direction, sims["block4"],
                 row["src_det"], row["dst_det"],
                 lock if lock is not None else "NA"))

    cols = ["handoff_frame", "direction", "src_cam", "dst_cam",
            "src_det", "dst_det", "lock_delay_frames"] + ["sim_" + b for b in BLOCKS]
    csv_path = os.path.join(out_dir, "handoff_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()
