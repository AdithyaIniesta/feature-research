"""Load a recorded session folder.

Key data model (learned from recorder.cpp + session inspection):
  * The recorder writes continuous raw_left.mkv / raw_right.mkv. The per-camera jpg
    folders (raw_left_frames/, raw_right_frames/) are extracted from those, numbered
    1..N per side.
  * events.jsonl `frame` is a REMAPPED GLOBAL counter (both cameras combined), so it
    does NOT index the per-side jpgs. Instead every event carries `t_ns` (nanoseconds
    since session start), which DOES map cleanly to a per-side jpg index by time.
  * The dense per-frame tracker box + confidence lives in ANGLE events:
        cam (1=left, 2=right), rect_x/rect_y/rect_w/rect_h, det (confidence).
    tracker.json and annotations.json are sparse (~5 entries) and largely unused.

Time -> jpg index:  idx = round(t_ns / t_max * (N_side - 1)) + 1, clamped to [1, N_side]
(assumes ~constant capture fps, verified accurate to ~1 frame on this data).
"""
import glob
import json
import os

CAM_SIDE = {1: "left", 2: "right"}


class Session:
    def __init__(self, path):
        self.path = path
        self.left_dir = os.path.join(path, "raw_left_frames")
        self.right_dir = os.path.join(path, "raw_right_frames")
        self.tracker = self._load_json("tracker.json", default={})
        self.annotations = self._load_json("annotations.json", default={})
        self.meta = self._load_json("meta.json", default={})
        self.events = self._load_events("events.jsonl")

        # per-side jpg counts
        self.n_left = self._count_frames(self.left_dir)
        self.n_right = self._count_frames(self.right_dir)

        # max timestamp across events, for the time->index map
        ts = [e.get("t_ns", 0) for e in self.events if e.get("t_ns")]
        self.t_max = max(ts) if ts else 1

    # --- loading ---------------------------------------------------------

    def _load_json(self, name, default=None):
        p = os.path.join(self.path, name)
        if not os.path.exists(p):
            return default
        with open(p, "r") as f:
            return json.load(f)

    def _load_events(self, name):
        p = os.path.join(self.path, name)
        out = []
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _count_frames(self, d):
        if not os.path.isdir(d):
            return 0
        return len(glob.glob(os.path.join(d, "*.jpg")))

    # --- time -> jpg index ----------------------------------------------

    def index_for_time(self, side, t_ns):
        n = self.n_left if side == "left" else self.n_right
        if n <= 1:
            return 1
        idx = int(round(t_ns / float(self.t_max) * (n - 1))) + 1
        return max(1, min(n, idx))

    def frame_path(self, side, idx):
        d = self.left_dir if side == "left" else self.right_dir
        return os.path.join(d, "%06d.jpg" % int(idx))

    def read_frame(self, side, idx):
        import cv2
        p = self.frame_path(side, idx)
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        return img

    def read_frame_at_time(self, side, t_ns):
        return self.read_frame(side, self.index_for_time(side, t_ns))

    # --- ANGLE events (dense boxes + confidence) -------------------------

    def angle_events(self, side=None):
        """Return list of dicts: {frame, t_ns, side, cam, box[xywh], det}."""
        out = []
        for e in self.events:
            if e.get("event") != "ANGLE":
                continue
            s = CAM_SIDE.get(e.get("cam"))
            if side is not None and s != side:
                continue
            if "rect_x" not in e:
                continue
            out.append({
                "frame": e.get("frame"),
                "t_ns": e.get("t_ns", 0),
                "side": s,
                "cam": e.get("cam"),
                "box": [e["rect_x"], e["rect_y"], e["rect_w"], e["rect_h"]],
                "det": e.get("det"),
            })
        out.sort(key=lambda r: r["t_ns"])
        return out

    def nearest_angle(self, side, t_ns, direction="any", max_dt_ns=None):
        """Closest ANGLE event on `side`. direction: 'before','after','any'."""
        best = None
        best_dt = None
        for a in self.angle_events(side):
            dt = a["t_ns"] - t_ns
            if direction == "before" and dt > 0:
                continue
            if direction == "after" and dt < 0:
                continue
            adt = abs(dt)
            if max_dt_ns is not None and adt > max_dt_ns:
                continue
            if best is None or adt < best_dt:
                best, best_dt = a, adt
        return best

    # --- event helpers ---------------------------------------------------

    def events_of(self, event_type):
        return [e for e in self.events if e.get("event") == event_type]

    def handoffs(self):
        return self.events_of("HANDOFF")

    def mode_changes(self):
        return self.events_of("MODE_CHANGE")
