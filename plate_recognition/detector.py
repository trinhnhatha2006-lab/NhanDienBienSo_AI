from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .types import BoundingBox, Recognition


class PlateDetector:
    """Bọc YOLO trong một lớp có thể tái sử dụng cho ảnh và từng khung video."""

    def __init__(self, model_path: str | Path, confidence: float = 0.5) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy model YOLO: {self.model_path}. "
                "Hãy đặt best.pt trong models/ hoặc dùng --model."
            )
        if not 0 < confidence <= 1:
            raise ValueError("confidence phải nằm trong khoảng (0, 1].")
        self.confidence = confidence
        self._model = YOLO(str(self.model_path))

    def find(self, frame: np.ndarray) -> list[Recognition]:
        """Phát hiện và cắt các biển số; luôn kẹp toạ độ trong ảnh."""
        height, width = frame.shape[:2]
        result = self._model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections: list[Recognition] = []

        if result.boxes is None:
            return detections

        coordinates = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(coordinates, scores, strict=True):
            left = max(0, min(width, int(x1)))
            top = max(0, min(height, int(y1)))
            right = max(0, min(width, int(x2)))
            bottom = max(0, min(height, int(y2)))
            if right <= left or bottom <= top:
                continue

            box = BoundingBox(left, top, right, bottom, float(score))
            detections.append(Recognition(box=box, crop=frame[top:bottom, left:right].copy()))

        return detections
