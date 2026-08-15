from __future__ import annotations

import re

import cv2
import easyocr
import numpy as np


class PlateTextReader:
    """Đọc ký tự biển số và chuẩn hoá về chữ-số viết hoa."""

    def __init__(self) -> None:
        # Khởi tạo một lần; EasyOCR sẽ dùng lại model cho cả video.
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, plate_image: np.ndarray) -> str:
        if plate_image.size == 0:
            return ""

        grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(grayscale, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        filtered = cv2.bilateralFilter(enlarged, 7, 50, 50)
        parts = self._reader.readtext(filtered, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
        raw_text = "".join(parts).upper()
        return re.sub(r"[^A-Z0-9.-]", "", raw_text)
