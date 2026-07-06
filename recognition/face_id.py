"""
recognition/face_id.py
======================
Face recognition module for JARVIS-XL.

How it works:
  • On startup  → scans webcam for ~5 s, tries to match against enrolled faces
  • On register → captures N frames, averages encodings, saves to faces/ directory
  • Matching    → cosine similarity against all enrolled face embeddings

Dependencies (auto-installed by core/installer.py):
  face-recognition   (wraps dlib's ResNet-based face model)
  opencv-python

Enrollment data is stored as NumPy .npy files:
  recognition/faces/<name>.npy  →  mean face encoding (128-dim vector)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np


class FaceIdentifier:
    THRESHOLD = 0.55      # cosine distance; lower = stricter match
    CAPTURE_N = 15        # frames to capture during registration
    SCAN_FRAMES = 20      # max frames to grab during identification scan

    def __init__(self, faces_dir: Path):
        self._dir = Path(faces_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._encodings: dict[str, np.ndarray] = {}
        self._load_all()

    # ── Load saved encodings ──────────────────────────────────────────────────
    def _load_all(self) -> None:
        for p in self._dir.glob("*.npy"):
            name = p.stem
            try:
                self._encodings[name] = np.load(str(p))
                print(f"[FaceID] Loaded encoding for '{name}'")
            except Exception as e:
                print(f"[FaceID] Could not load {p}: {e}")

    # ── Identify ──────────────────────────────────────────────────────────────
    def identify(self, timeout: float = 5.0) -> tuple[Optional[str], float]:
        """
        Open the webcam, grab up to SCAN_FRAMES frames, and attempt to match
        any detected face against enrolled profiles.

        Returns (name, confidence) or (None, 0.0) if no match.
        Confidence is 1 - cosine_distance, scaled to [0, 1].
        """
        if not self._encodings:
            return None, 0.0

        try:
            import face_recognition
            import cv2
        except ImportError:
            print("[FaceID] face_recognition or cv2 not installed — skipping face scan.")
            return None, 0.0

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("[FaceID] No camera available.")
            return None, 0.0

        best_name  = None
        best_conf  = 0.0
        deadline   = time.time() + timeout
        frames_checked = 0

        try:
            while time.time() < deadline and frames_checked < self.SCAN_FRAMES:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                frames_checked += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="hog")
                if not locs:
                    continue

                encs = face_recognition.face_encodings(rgb, locs)
                for enc in encs:
                    name, conf = self._match(enc)
                    if conf > best_conf:
                        best_conf = conf
                        best_name = name

                if best_conf >= (1 - self.THRESHOLD):
                    break  # confident early exit
        finally:
            cam.release()

        if best_conf < (1 - self.THRESHOLD):
            return None, 0.0
        return best_name, best_conf

    # ── Register ──────────────────────────────────────────────────────────────
    def register(self, name: str) -> bool:
        """
        Capture CAPTURE_N frames from the webcam and save the mean face encoding
        to recognition/faces/<name>.npy.
        """
        try:
            import face_recognition
            import cv2
        except ImportError:
            print("[FaceID] face_recognition or cv2 not installed.")
            return False

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return False

        collected = []
        print(f"[FaceID] Capturing {self.CAPTURE_N} frames for '{name}'…")

        try:
            while len(collected) < self.CAPTURE_N:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="hog")
                if not locs:
                    continue
                encs = face_recognition.face_encodings(rgb, locs)
                if encs:
                    collected.append(encs[0])
        finally:
            cam.release()

        if not collected:
            return False

        mean_enc = np.mean(collected, axis=0)
        out_path = self._dir / f"{name}.npy"
        np.save(str(out_path), mean_enc)
        self._encodings[name] = mean_enc
        print(f"[FaceID] Registered '{name}' ({len(collected)} frames).")
        return True

    # ── Matching helper ───────────────────────────────────────────────────────
    def _match(self, encoding: np.ndarray) -> tuple[Optional[str], float]:
        """Return (best_name, confidence) where confidence = 1 - L2_dist / 2."""
        best_name = None
        best_dist = 1.0
        for name, ref in self._encodings.items():
            dist = float(np.linalg.norm(encoding - ref))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        confidence = max(0.0, 1.0 - best_dist / 2.0)
        return best_name, confidence

    # ── Utility ───────────────────────────────────────────────────────────────
    def list_users(self) -> list[str]:
        return list(self._encodings.keys())

    def delete_user(self, name: str) -> bool:
        p = self._dir / f"{name}.npy"
        if p.exists():
            p.unlink()
            self._encodings.pop(name, None)
            return True
        return False
