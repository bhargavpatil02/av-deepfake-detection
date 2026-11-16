"""
Module 3 - Visual / lip feature extraction.

Two detection backends are supported and the choice is reported honestly:

* "mediapipe" - FaceMesh gives 468 landmarks; we use a small, well-known set
  around the lips to measure mouth opening, mouth width and lip-corner motion.
* "opencv_haar" - a Haar cascade finds a face box only. Without landmarks we
  cannot measure mouth opening; we can only measure frame-to-frame pixel
  change inside an approximate mouth region. Features that need landmarks are
  left as `None` (never fabricated).

All geometric features are scale invariant (divided by face width/height or by
frame size), so a close-up and a wide shot of the same face are comparable.

IMPORTANT LIMITATION: sampled frames are ~1/12 s apart, so "motion" is the
change between analysed frames, not true optical flow. This is a coarse proxy
for articulation, adequate for a baseline but not for phoneme-level analysis.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

from . import config

# FaceMesh landmark ids around the mouth (iBUG 68-style indices used by MediaPipe)
LIP_UPPER_OUT, LIP_LOWER_OUT = 13, 14     # vertical opening of the outer lips
LIP_UPPER_IN, LIP_LOWER_IN = 78, 308      # vertical opening of the inner lips
LIP_LEFT, LIP_RIGHT = 61, 291             # mouth corners

# Mouth crops are resized to this fixed patch before frame differencing, so the
# comparison is valid even though the landmark box changes size every frame.
MOUTH_PATCH_W, MOUTH_PATCH_H = 64, 32

VISUAL_FEATURE_NAMES = [
    "mean_mar", "std_mar",
    "mean_mouth_open_norm", "std_mouth_open_norm", "p95_mouth_open_norm",
    "mean_mouth_width_norm", "std_mouth_width_norm",
    "mean_abs_dmar", "std_abs_dmar", "p95_abs_dmar",
    "mean_mouth_pixel_motion", "std_mouth_pixel_motion",
    "mean_lipcorner_motion_norm",
    "face_detect_rate", "n_frames_detected", "mean_face_area_frac",
]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


class FaceAnalyzer:
    """Per-frame face analysis with a graceful backend fallback."""

    def __init__(self, backend: str = "auto", work_width: int = config.VISUAL_WORK_WIDTH):
        self.work_width = work_width
        self.backend = "unavailable"
        self._mesh = None          # MediaPipe FaceMesh (legacy) or FaceLandmarker (Tasks)
        self._mp_api: str | None = None
        self._cascade = None
        self.notes: list[str] = []

        # Work on a private copy of the environment so OpenCV prints no warnings.
        self._silence = contextlib.redirect_stderr(open(_nul_device(), "w"))

        if backend in ("auto", "mediapipe"):
            self._try_mediapipe()
        if self.backend == "unavailable" and backend in ("auto", "opencv_haar"):
            self._try_haar()

    # ------------------------------------------------------------ backends
    def _try_mediapipe(self) -> None:
        import mediapipe as mp

        # 1) Legacy Solutions API (mediapipe <= 0.10.x)
        if hasattr(mp, "solutions"):
            try:
                with self._silence:
                    face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                    )
                self._mesh = face_mesh
                self._mp_api = "solutions"
                self.backend = "mediapipe"
                self.notes.append("MediaPipe FaceMesh (Solutions API) backend active.")
                return
            except Exception as exc:
                self.notes.append(
                    f"MediaPipe Solutions API failed ({type(exc).__name__}); trying Tasks."
                )

        # 2) Tasks API (mediapipe >= 0.10.9 / 1.x)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = _ensure_landmarker_model(self.notes)
            if model_path is None:
                return

            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(opts)
            self._mp_api = "tasks"
            self.backend = "mediapipe"
            self.notes.append("MediaPipe FaceLandmarker (Tasks API) backend active.")
        except Exception as exc:  # not installed, download failed, or API change
            self.notes.append(f"MediaPipe unavailable ({type(exc).__name__}); trying Haar.")

    def _try_haar(self) -> None:
        try:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(path))
            if cascade.empty():
                raise RuntimeError("cascade file could not be loaded")
            self._cascade = cascade
            self.backend = "opencv_haar"
            self.notes.append(
                "OpenCV Haar cascade backend active: face box only, no lip landmarks."
            )
        except Exception as exc:
            self.notes.append(f"Haar cascade unavailable ({exc}).")
            self.backend = "unavailable"

    # -------------------------------------------------------------- helpers
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > self.work_width:
            s = self.work_width / w
            frame = cv2.resize(frame, (self.work_width, max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
        return frame

    def _analyze_mediapipe(self, frame: np.ndarray) -> dict | None:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._mp_api == "tasks":
            import mediapipe as mp

            with self._silence:
                res = self._mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                return None
            lm = res.face_landmarks[0]
        else:
            with self._silence:
                res = self._mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float64)

        left, right = pt(LIP_LEFT), pt(LIP_RIGHT)
        up_out, lo_out = pt(LIP_UPPER_OUT), pt(LIP_LOWER_OUT)
        up_in, lo_in = pt(LIP_UPPER_IN), pt(LIP_LOWER_IN)

        mouth_w = max(_dist(left, right), 1e-6)
        open_out = _dist(up_out, lo_out)
        open_in = _dist(up_in, lo_in)

        xs = np.array([l.x for l in lm]) * w
        ys = np.array([l.y for l in lm]) * h
        face_w = max(float(xs.max() - xs.min()), 1e-6)
        face_h = max(float(ys.max() - ys.min()), 1e-6)

        return {
            "frame_size": (h, w),
            "mar": open_out / mouth_w,                       # mouth aspect ratio
            "mouth_open_norm": open_in / face_h,             # inner opening / face height
            "mouth_width_norm": mouth_w / face_w,            # mouth width / face width
            "face_area_frac": (face_w * face_h) / float(w * h),
            "mouth_box": (int(min(left[0], right[0])), int(min(up_out[1], lo_out[1])),
                          int(max(left[0], right[0])), int(max(up_out[1], lo_out[1]))),
            "corners": (left, right),
        }

    def _analyze_haar(self, frame: np.ndarray) -> dict | None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._silence:
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                   minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        # Approximate mouth region: lower-middle third of the face box.
        mx0, mx1 = int(x + 0.20 * fw), int(x + 0.80 * fw)
        my0, my1 = int(y + 0.60 * fh), int(y + 0.92 * fh)
        return {
            "frame_size": (h, w),
            "mar": None,                 # no landmarks -> no honest measurement
            "mouth_open_norm": None,
            "mouth_width_norm": None,
            "face_area_frac": (fw * fh) / float(w * h),
            "mouth_box": (mx0, my0, mx1, my1),
            "corners": None,
        }

    # ---------------------------------------------------------------- public
    def analyze(self, frame_bgr: np.ndarray) -> dict | None:
        frame = self._resize(frame_bgr)
        if self.backend == "mediapipe":
            return self._analyze_mediapipe(frame)
        if self.backend == "opencv_haar":
            return self._analyze_haar(frame)
        return None

    def close(self) -> None:
        if self._mesh is not None:
            with contextlib.suppress(Exception):
                self._mesh.close()
        with contextlib.suppress(Exception):
            self._silence.close()


def _nul_device() -> str:
    import os

    return os.devnull


def _ensure_landmarker_model(notes: list[str]) -> Path | None:
    """Return the Tasks face-landmarker model, downloading it once if needed."""
    path = Path(config.FACE_LANDMARKER_MODEL)
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    try:
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        notes.append("Downloading MediaPipe face-landmarker model (~3.6 MB, once)...")
        urllib.request.urlretrieve(config.FACE_LANDMARKER_URL, tmp)
        tmp.replace(path)
        return path
    except Exception as exc:
        notes.append(f"Face-landmarker model download failed ({type(exc).__name__}).")
        return None


def _safe_stats(values: list[float | None]) -> tuple[float | None, float | None, float | None]:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, None
    return float(arr.mean()), float(arr.std()), float(np.percentile(arr, 95))


def extract_visual_features(
    frames, backend: str = "auto", progress=None
) -> dict:
    """
    Walk a stream of (timestamp, frame) pairs and build visual features.

    `frames` may be any iterable (we stream; nothing is stored).
    Returns a dict with `features` (None for unmeasurable values), the
    per-frame series needed for plots/sync, and diagnostics.
    """
    analyzer = FaceAnalyzer(backend=backend)
    if analyzer.backend == "unavailable":
        return {
            "ok": False,
            "error": "No face-detection backend available (install mediapipe or opencv).",
            "notes": analyzer.notes,
            "features": {n: None for n in VISUAL_FEATURE_NAMES},
        }

    records: list[dict] = []
    total = 0
    prev_mouth = None
    prev_corners = None
    prev_t = None
    pixel_motions: list[float] = []
    corner_motions: list[float] = []
    dmar: list[float] = []

    for t, frame in frames:
        total += 1
        cur_motion = None
        cur_corner = None
        try:
            info = analyzer.analyze(frame)
        except Exception:
            info = None
        if info is None:
            prev_mouth, prev_corners, prev_t = None, None, None
            records.append({"t": t, "detected": False})
            if progress and total % 20 == 0:
                progress(total)
            continue

        mar = info["mar"]
        if mar is not None and records:
            last = next((r for r in reversed(records) if r["detected"]), None)
            if last is not None and last.get("mar") is not None:
                dmar.append(abs(mar - last["mar"]))

        # Frame-to-frame pixel motion inside the mouth region (backend agnostic).
        x0, y0, x1, y1 = info["mouth_box"]
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(frame.shape[1], max(x0 + 1, x1))
        y1 = min(frame.shape[0], max(y0 + 1, y1))
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        mouth_crop = None
        if gray.size:
            # Fixed patch size -> every frame is directly comparable.
            mouth_crop = cv2.resize(gray, (MOUTH_PATCH_W, MOUTH_PATCH_H),
                                    interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev_mouth is not None and prev_t is not None and mouth_crop is not None and \
                prev_mouth.shape == mouth_crop.shape:
            dt = max(t - prev_t, 1e-3)
            cur_motion = float(np.mean(np.abs(mouth_crop - prev_mouth)) / dt)
            pixel_motions.append(cur_motion)
        prev_mouth, prev_t = (mouth_crop.copy() if mouth_crop.size else None), t

        # Lip-corner displacement (landmark backend only).
        if info["corners"] is not None:
            if prev_corners is not None:
                face_w = max(
                    1e-6,
                    np.hypot(*(info["corners"][0] - info["corners"][1]))
                    + 1e-6,
                )
                disp = (np.hypot(*(info["corners"][0] - prev_corners[0])) +
                        np.hypot(*(info["corners"][1] - prev_corners[1]))) / 2.0
                cur_corner = float(disp / face_w)
                corner_motions.append(cur_corner)
            prev_corners = info["corners"]

        records.append({"t": t, "detected": True, "mar": mar,
                        "mouth_open_norm": info["mouth_open_norm"],
                        "mouth_width_norm": info["mouth_width_norm"],
                        "face_area_frac": info["face_area_frac"],
                        "motion": cur_motion, "corner_motion": cur_corner})
        if progress and total % 20 == 0:
            progress(total)

    analyzer.close()

    detected = [r for r in records if r["detected"]]
    n_det = len(detected)
    rate = n_det / total if total else 0.0
    feats: dict[str, float | None] = {n: None for n in VISUAL_FEATURE_NAMES}
    feats["face_detect_rate"] = float(rate)
    feats["n_frames_detected"] = float(n_det)

    m_mean, m_std, m_p95 = _safe_stats([r["mar"] for r in detected])
    o_mean, o_std, o_p95 = _safe_stats([r["mouth_open_norm"] for r in detected])
    w_mean, w_std, _ = _safe_stats([r["mouth_width_norm"] for r in detected])
    d_mean, d_std, d_p95 = _safe_stats(dmar)
    p_mean, p_std, _ = _safe_stats(pixel_motions)
    c_mean, _, _ = _safe_stats(corner_motions)
    a_mean, _, _ = _safe_stats([r["face_area_frac"] for r in detected])

    feats.update({
        "mean_mar": m_mean, "std_mar": m_std,
        "mean_mouth_open_norm": o_mean, "std_mouth_open_norm": o_std,
        "p95_mouth_open_norm": o_p95,
        "mean_mouth_width_norm": w_mean, "std_mouth_width_norm": w_std,
        "mean_abs_dmar": d_mean, "std_abs_dmar": d_std, "p95_abs_dmar": d_p95,
        "mean_mouth_pixel_motion": p_mean, "std_mouth_pixel_motion": p_std,
        "mean_lipcorner_motion_norm": c_mean,
        "mean_face_area_frac": a_mean,
    })

    usable = n_det >= config.MIN_FACE_FRAMES and rate > 0
    notes = list(analyzer.notes)
    if not usable:
        notes.append(
            f"Face found in only {n_det}/{total} analysed frames "
            f"(minimum {config.MIN_FACE_FRAMES}); visual features unusable."
        )
    elif analyzer.backend == "opencv_haar":
        notes.append("Landmark-based mouth features unavailable on the Haar backend.")

    return {
        "ok": usable,
        "backend": analyzer.backend,
        "features": feats,
        "names": VISUAL_FEATURE_NAMES,
        "n_frames_analysed": total,
        "n_frames_detected": n_det,
        "face_detect_rate": rate,
        "records": records,
        "notes": notes,
        "error": None if usable else "No usable face/landmark detections.",
    }
