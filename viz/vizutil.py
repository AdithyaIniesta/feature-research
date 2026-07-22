"""Shared drawing helpers for the replay videos."""
import cv2
import numpy as np


def ncc(a_bgr, b_bgr, size=64):
    """Normalized cross-correlation (classic pixel template match) in [0,1].

    Both crops are resized to size x size, grayscaled, zero-meaned and unit-normed;
    the dot product is the NCC. This is the 'old-style RGB template' baseline.
    """
    def prep(c):
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g = cv2.resize(g, (size, size))
        g -= g.mean()
        n = np.linalg.norm(g)
        return g / n if n > 0 else g
    va, vb = prep(a_bgr), prep(b_bgr)
    val = float((va * vb).sum())  # in [-1, 1]
    return max(0.0, val)          # clamp to [0,1] for the meter


def draw_meter(img, x, y, w, h, value, label, color):
    """Draw a labeled horizontal bar meter. value in [0,1]."""
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fillw = int(w * value)
    cv2.rectangle(img, (x, y), (x + fillw, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(img, "%s %.2f" % (label, value), (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_box(img, box_xywh, color=(0, 255, 0), label=None):
    x, y, w, h = [int(round(v)) for v in box_xywh]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(img, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def paste_thumb(img, crop, x, y, box=96, label=None):
    """Paste a small thumbnail of `crop` at (x,y) with an optional caption."""
    if crop is None or crop.size == 0:
        return
    th = cv2.resize(crop, (box, box))
    h, w = img.shape[:2]
    if y + box <= h and x + box <= w:
        img[y:y + box, x:x + box] = th
        cv2.rectangle(img, (x, y), (x + box, y + box), (200, 200, 200), 1)
        if label:
            cv2.putText(img, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def open_writer(path, fps, w, h):
    """Try mp4 first, fall back to MJPG .avi if the codec is unavailable."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if vw.isOpened():
        return vw, path
    alt = path.rsplit(".", 1)[0] + ".avi"
    vw = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    return vw, alt
