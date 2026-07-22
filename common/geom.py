"""Box geometry helpers.

Two box conventions are used in the data:
  tracker.json     -> [x, y, w, h]      (top-left + size)
  annotations.json -> [x1, y1, x2, y2]  (corners)
"""


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]


def iou(a_xyxy, b_xyxy):
    ax1, ay1, ax2, ay2 = a_xyxy
    bx1, by1, bx2, by2 = b_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def crop(img, box_xywh, pad=0.0):
    """Crop [x,y,w,h] from an image, clamped to bounds. pad = fractional margin."""
    h_img, w_img = img.shape[:2]
    x, y, w, h = box_xywh
    if pad:
        x -= w * pad; y -= h * pad
        w += 2 * w * pad; h += 2 * h * pad
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(w_img, int(round(x + w)))
    y2 = min(h_img, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]
