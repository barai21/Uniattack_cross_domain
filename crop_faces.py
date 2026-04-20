"""
Face Crop Preprocessing – MediaPipe (all versions) + OpenCV fallback
=====================================================================
Reads Protocol 2.2 txt files, detects + crops faces, saves to UniAttackData_P/.

Detector priority (auto-selected at startup):
  1. MediaPipe Tasks API  (mediapipe >= 0.10)  ← new style
  2. MediaPipe Solutions  (mediapipe < 0.10)   ← old style
  3. OpenCV Haar Cascade  (always available)   ← no-install fallback

If no face is found at all → centre-crop fallback (no missing files).

Also writes *_cropped.txt files pointing to UniAttackData_P/ paths.

Install:
  pip install mediapipe opencv-python tqdm
"""

import os
import sys
import cv2
import numpy as np
import urllib.request
from tqdm import tqdm

# ─────────────────────────── CONFIG ───────────────────────────────────────────
BASE_DIR  = r"C:\Users\bhask\Documents\Project\Resnet_50"
SRC_ROOT  = os.path.join(BASE_DIR, "UniAttackData")
DST_ROOT  = os.path.join(BASE_DIR, "UniAttackData_P")

TXT_FILES = [
    os.path.join(BASE_DIR, r"path\UniAttackData@p2.2_image_train.txt"),
    os.path.join(BASE_DIR, r"path\UniAttackData@p2.2_image_dev.txt"),
    os.path.join(BASE_DIR, r"path\UniAttackData@p2.2_image_test.txt"),
]

CROP_SIZE        = 224    # output image size (square)
MARGIN_RATIO     = 0.30   # expand face box by 30% each side
MIN_CONF         = 0.5    # primary detection threshold
LOW_CONF         = 0.3    # retry threshold
JPEG_QUALITY     = 95
REPORT_EVERY     = 500


# ─────────────────────────── DETECTOR ─────────────────────────────────────────
class FaceDetector:
    """
    Wraps three backends in priority order.
    Call detect(img_bgr) -> list of (x1,y1,x2,y2) pixel boxes.
    """

    def __init__(self):
        self.backend = None
        self._init()

    def _init(self):
        # ── 1. MediaPipe Tasks (>= 0.10) ──────────────────────────────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")
            if not os.path.exists(model_path):
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_detector/blaze_face_short_range/float16/latest/"
                    "blaze_face_short_range.tflite"
                )
                print(f"[INFO] Downloading MediaPipe face model → {model_path}")
                urllib.request.urlretrieve(url, model_path)

            def _make(conf):
                opts = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    min_detection_confidence=conf,
                )
                return mp_vision.FaceDetector.create_from_options(opts)

            self._det_hi  = _make(MIN_CONF)
            self._det_lo  = _make(LOW_CONF)
            self._mp      = mp
            self.backend  = "mediapipe_tasks"
            print("[INFO] Face detector: MediaPipe Tasks API (>= 0.10)")
            return
        except Exception:
            pass

        # ── 2. MediaPipe Solutions (< 0.10) ───────────────────────────────────
        try:
            import mediapipe as mp
            self._det_hi = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=MIN_CONF)
            self._det_lo = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=LOW_CONF)
            self.backend = "mediapipe_solutions"
            print("[INFO] Face detector: MediaPipe Solutions API (< 0.10)")
            return
        except Exception:
            pass

        # ── 3. OpenCV Haar Cascade ────────────────────────────────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self.backend  = "opencv_haar"
        print("[INFO] Face detector: OpenCV Haar Cascade (fallback)")

    def detect(self, img_bgr):
        """Returns list of (x1,y1,x2,y2) in pixel coords, or []."""
        h, w = img_bgr.shape[:2]

        if self.backend == "mediapipe_tasks":
            return self._detect_tasks(img_bgr, h, w)
        elif self.backend == "mediapipe_solutions":
            return self._detect_solutions(img_bgr, h, w)
        else:
            return self._detect_haar(img_bgr)

    def _detect_tasks(self, img_bgr, h, w):
        import mediapipe as mp
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = self._det_hi.detect(mp_img)
        if not result.detections:
            result = self._det_lo.detect(mp_img)
        boxes = []
        for det in result.detections:
            bb = det.bounding_box
            boxes.append((bb.origin_x, bb.origin_y,
                           bb.origin_x + bb.width,
                           bb.origin_y + bb.height))
        return boxes

    def _detect_solutions(self, img_bgr, h, w):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res     = self._det_hi.process(img_rgb)
        if not res.detections:
            res = self._det_lo.process(img_rgb)
        boxes = []
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1 = int(bb.xmin * w)
                y1 = int(bb.ymin * h)
                x2 = int((bb.xmin + bb.width)  * w)
                y2 = int((bb.ymin + bb.height) * h)
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _detect_haar(self, img_bgr):
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) == 0:
            return []
        return [(x, y, x + bw, y + bh) for (x, y, bw, bh) in faces]


# ─────────────────────────── CROP HELPERS ─────────────────────────────────────
def make_crop(img_bgr, boxes, margin=MARGIN_RATIO):
    """
    Given detection boxes, pick the largest, expand by margin,
    return (crop, True). Returns (None, False) if boxes is empty.
    """
    if not boxes:
        return None, False

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw // 2, y1 + bh // 2
    side   = max(bw, bh)
    half   = int(side * (1 + margin) / 2)

    x1c = max(cx - half, 0)
    y1c = max(cy - half, 0)
    x2c = min(cx + half, w)
    y2c = min(cy + half, h)

    crop = img_bgr[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, False

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR), True


def centre_crop(img_bgr):
    """Fallback: centred square crop → CROP_SIZE."""
    h, w  = img_bgr.shape[:2]
    side  = min(h, w)
    cy, cx = h // 2, w // 2
    half  = side // 2
    crop  = img_bgr[cy-half:cy+half, cx-half:cx+half]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────── PATH HELPERS ─────────────────────────────────────
def src_path(rel):
    return os.path.join(BASE_DIR, rel.replace("/", os.sep))


def dst_path(rel):
    # UniAttackData/Data/... → UniAttackData_P/Data/...
    stripped = rel.replace("UniAttackData/", "", 1)
    path     = os.path.join(DST_ROOT, stripped.replace("/", os.sep))
    return os.path.splitext(path)[0] + ".jpg"


# ─────────────────────────── COLLECT ENTRIES ──────────────────────────────────
def collect_entries(txt_files):
    seen, entries = set(), []
    for txt in txt_files:
        if not os.path.exists(txt):
            print(f"[WARN] Not found, skipping: {txt}")
            continue
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel, label = parts[0], int(parts[1])
                if rel not in seen:
                    seen.add(rel)
                    entries.append((rel, label))
    return entries


# ─────────────────────────── MAIN CROP LOOP ───────────────────────────────────
def run(entries, detector):
    total    = len(entries)
    saved    = 0
    fallback = 0
    skipped  = 0

    print(f"\n[INFO] Images to process : {total}")
    print(f"[INFO] Output root       : {DST_ROOT}")
    print(f"[INFO] Crop size         : {CROP_SIZE}x{CROP_SIZE}  |  Margin: {MARGIN_RATIO*100:.0f}%\n")

    for i, (rel, label) in enumerate(tqdm(entries, desc="Cropping", unit="img")):
        src = src_path(rel)
        dst = dst_path(rel)

        if not os.path.exists(src):
            skipped += 1
            continue

        img_bgr = cv2.imread(src)
        if img_bgr is None:
            skipped += 1
            continue

        boxes       = detector.detect(img_bgr)
        crop, found = make_crop(img_bgr, boxes)

        if crop is None:
            crop     = centre_crop(img_bgr)
            fallback += 1

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1

        if (i + 1) % REPORT_EVERY == 0:
            detected = saved - fallback
            print(
                f"\n  [{i+1}/{total}]  saved={saved}  "
                f"face_detected={detected} ({detected/max(saved,1)*100:.1f}%)  "
                f"centre_fallback={fallback}  skipped={skipped}"
            )

    detected = saved - fallback
    print("\n" + "=" * 60)
    print("  FACE CROP COMPLETE")
    print("=" * 60)
    print(f"  Total unique images   : {total}")
    print(f"  Saved                 : {saved}")
    print(f"  Face detected         : {detected}  ({detected/max(saved,1)*100:.1f}%)")
    print(f"  Centre-crop fallback  : {fallback}  ({fallback/max(saved,1)*100:.1f}%)")
    print(f"  Skipped (src missing) : {skipped}")
    print(f"  Output directory      : {DST_ROOT}")
    print("=" * 60)


# ─────────────────────────── WRITE CROPPED TXT ────────────────────────────────
def write_cropped_txts(txt_files):
    print("\n[INFO] Writing *_cropped.txt files ...")
    for txt in txt_files:
        if not os.path.exists(txt):
            continue
        lines = []
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts   = line.split()
                rel     = parts[0]
                label   = parts[1]
                stripped = rel.replace("UniAttackData/", "", 1)
                new_rel  = "UniAttackData_P/" + stripped
                new_rel  = os.path.splitext(new_rel)[0] + ".jpg"
                new_rel  = new_rel.replace("\\", "/")
                lines.append(f"{new_rel} {label}")

        out = os.path.splitext(txt)[0] + "_cropped.txt"
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved: {out}  ({len(lines)} entries)")


# ─────────────────────────── ENTRY POINT ──────────────────────────────────────
def main():
    print("=" * 60)
    print("  MediaPipe Face Crop  –  Protocol 2.2")
    print("=" * 60)

    detector = FaceDetector()
    entries  = collect_entries(TXT_FILES)

    if not entries:
        print("[ERROR] No entries found. Check TXT_FILES paths.")
        sys.exit(1)

    run(entries, detector)
    write_cropped_txts(TXT_FILES)

    print("\n[DONE] Update your training scripts:")
    print(f"  DATA_ROOT = r\"{DST_ROOT}\"")
    print("  Use the *_cropped.txt files instead of the originals.")


if __name__ == "__main__":
    main()
