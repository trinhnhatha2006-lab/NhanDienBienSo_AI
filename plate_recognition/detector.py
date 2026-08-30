from __future__ import annotations

import hashlib
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
        self.model_sha256 = self._sha256(self.model_path)
        self._model = YOLO(str(self.model_path))
        names = self._model.names
        self.class_names = dict(enumerate(names)) if isinstance(names, list) else dict(names)
        matching_classes = [
            class_id
            for class_id, name in self.class_names.items()
            if str(name).strip().lower() in {"license_plate", "license plate"}
        ]
        if matching_classes:
            self.class_id = int(matching_classes[0])
        elif len(self.class_names) == 1:
            self.class_id = int(next(iter(self.class_names)))
        else:
            raise ValueError(
                "Model có nhiều class nhưng không có class 'license_plate'."
            )

    def find(self, frame: np.ndarray) -> list[Recognition]:
        """Phát hiện và cắt các biển số; luôn kẹp toạ độ trong ảnh."""
        height, width = frame.shape[:2]
        result = self._model.predict(
            frame,
            conf=self.confidence,
            classes=[self.class_id],
            verbose=False,
        )[0]
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

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
