from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RunStorage:
    """Quản lý toàn bộ artefact của một lượt nhận diện, không ghi đè lần chạy trước."""

    def __init__(self, root: str | Path = "runs") -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(root) / stamp
        self.plates_dir = self.run_dir / "plates"
        self.plates_dir.mkdir(parents=True, exist_ok=False)
        self._plate_number = 0

    def save_plate(self, image: np.ndarray) -> Path:
        self._plate_number += 1
        destination = self.plates_dir / f"plate_{self._plate_number:05d}.jpg"
        self._write_image(destination, image)
        return destination

    def save_image(self, image: np.ndarray) -> Path:
        destination = self.run_dir / "annotated_image.jpg"
        self._write_image(destination, image)
        return destination

    def save_report(self, report: dict[str, Any]) -> Path:
        destination = self.run_dir / "recognitions.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    @staticmethod
    def _write_image(destination: Path, image: np.ndarray) -> None:
        # imencode giúp OpenCV ghi được đường dẫn có dấu trên Windows.
        ok, encoded = cv2.imencode(destination.suffix, image)
        if not ok:
            raise OSError(f"Không thể lưu ảnh: {destination}")
        encoded.tofile(str(destination))
