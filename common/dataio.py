"""Load a recorded session folder.

Layout expected (one session = one folder):
  raw_left_frames/000001.jpg ...
  raw_right_frames/000001.jpg ...
  events.jsonl        (START, CMD, HANDOFF, ANGLE, MODE_CHANGE, SHUTDOWN)
  tracker.json        { "<frame>": {"frame":N, "left":[x,y,w,h], "right":[x,y,w,h]} }
  annotations.json    { "<frame>": {"frame":N, "left":[x1,y1,x2,y2], "right":[...]} }
  meta.json
"""
import json
import os


class Session:
    def __init__(self, path):
        self.path = path
        self.left_dir = os.path.join(path, "raw_left_frames")
        self.right_dir = os.path.join(path, "raw_right_frames")
        self.tracker = self._load_json("tracker.json")
        self.annotations = self._load_json("annotations.json")
        self.meta = self._load_json("meta.json")
        self.events = self._load_events("events.jsonl")

    def _load_json(self, name):
        p = os.path.join(self.path, name)
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

    # --- convenience accessors -------------------------------------------

    def frame_path(self, cam, frame):
        """cam in {'left','right'}, frame is an int -> jpg path."""
        d = self.left_dir if cam == "left" else self.right_dir
        return os.path.join(d, "%06d.jpg" % int(frame))

    def read_frame(self, cam, frame):
        import cv2
        p = self.frame_path(cam, frame)
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        return img

    def tracker_box(self, cam, frame):
        """Return [x,y,w,h] or None."""
        rec = self.tracker.get(str(frame))
        if rec is None:
            return None
        return rec.get(cam)

    def annotation_box(self, cam, frame):
        """Return [x1,y1,x2,y2] or None."""
        rec = self.annotations.get(str(frame))
        if rec is None:
            return None
        return rec.get(cam)

    def events_of(self, event_type):
        return [e for e in self.events if e.get("event") == event_type]

    def handoffs(self):
        return self.events_of("HANDOFF")

    def mode_changes(self):
        return self.events_of("MODE_CHANGE")
