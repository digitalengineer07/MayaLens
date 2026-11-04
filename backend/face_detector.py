import os
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    prob: float = 1.0
    landmarks: Optional[np.ndarray] = None  # (5, 2): re, le, nose, r-mouth, l-mouth

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


class MayaLensFaceDetector:
    """
    High-accuracy face detector and landmark alignment for MayaLens multi-stream inference.
    Uses OpenCV YuNet deep neural detector (or fallback contour/skin analysis).
    """
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.6):
        self.conf_threshold = conf_threshold
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "models" / "face_detection_yunet.onnx")
        self.model_path = model_path
        self._detector = None
        self._last_size = (0, 0)
        
        if os.path.exists(self.model_path) and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                self._detector = cv2.FaceDetectorYN_create(
                    self.model_path, "", (320, 320), conf_threshold, 0.3, 5000
                )
            except Exception as e:
                print(f"[MayaLensFaceDetector] Init warning: {e}")

    def detect_faces(self, frame_bgr: np.ndarray) -> List[BoundingBox]:
        h, w = frame_bgr.shape[:2]
        if self._detector is not None:
            try:
                if self._last_size != (w, h):
                    self._detector.setInputSize((w, h))
                    self._last_size = (w, h)
                
                _, faces = self._detector.detect(frame_bgr)
                if faces is not None and len(faces) > 0:
                    results = []
                    for f in faces:
                        score = float(f[14])
                        if score >= self.conf_threshold:
                            x, y, bw, bh = f[0], f[1], f[2], f[3]
                            lms = f[4:14].reshape((5, 2))
                            results.append(BoundingBox(
                                x1=float(x),
                                y1=float(y),
                                x2=float(x + bw),
                                y2=float(y + bh),
                                prob=score,
                                landmarks=lms
                            ))
                    if results:
                        # Return largest faces first
                        return sorted(results, key=lambda b: -(b.w * b.h))
            except Exception as e:
                print(f"[MayaLensFaceDetector] Detection error: {e}")

        # Fallback: check skin color region in YCrCb
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) & (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
        skin_ratio = np.mean(skin_mask)
        if skin_ratio > 0.05:
            # Estimate bounding box around major skin cluster
            contours, _ = cv2.findContours(skin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                if bw > 30 and bh > 30:
                    return [BoundingBox(float(x), float(y), float(x + bw), float(y + bh), prob=0.6)]

        return []

    def align_face(
        self, 
        frame_rgb: np.ndarray, 
        box: BoundingBox, 
        scale: float = 1.15, 
        out_size: int = 224
    ) -> np.ndarray:
        """
        Aligns facial region by eye-line angle (if landmarks present)
        and crops with requested multi-scale factor (1.15x tight vs 1.70x context).
        """
        h, w = frame_rgb.shape[:2]
        
        if box.landmarks is not None and len(box.landmarks) >= 2:
            re_, le = box.landmarks[0], box.landmarks[1]
            angle = math.degrees(math.atan2(le[1] - re_[1], le[0] - re_[0]))
            M = cv2.getRotationMatrix2D((box.cx, box.cy), angle, 1.0)
            rot = cv2.warpAffine(frame_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            rot = frame_rgb

        side = max(int(round(max(box.w, box.h) * scale)), 16)
        patch = cv2.getRectSubPix(rot, (side, side), (box.cx, box.cy))
        
        if patch is None or patch.size == 0:
            return cv2.resize(frame_rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)

        interp = cv2.INTER_AREA if side > out_size else cv2.INTER_CUBIC
        return cv2.resize(patch, (out_size, out_size), interpolation=interp)

