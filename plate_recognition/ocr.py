from __future__ import annotations

import re

import cv2
import easyocr
import numpy as np

from .types import OCRResult


ALLOWED_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."


def normalize_plate_text(raw_text: str) -> str:
    """Làm sạch và chỉ định dạng khi chuỗi khớp chắc chắn biển số Việt Nam."""
    upper_text = raw_text.upper()
    # Biển xe máy đời cũ có thể có 4 số ở dòng dưới, ví dụ 54-U5 / 7001.
    # Chuỗi compact của kiểu này trùng hình dạng với biển ô tô 5 số, nên chỉ
    # định dạng khi OCR còn giữ được dấu ngăn cách và khoảng trắng giữa 2 dòng.
    old_motorcycle_match = re.fullmatch(
        r"\s*(\d{2})\s*[-.]\s*([A-Z]\d)\s+(\d{4})\s*",
        upper_text,
    )
    if old_motorcycle_match:
        province, series, number = old_motorcycle_match.groups()
        return f"{province}-{series} {number}"

    compact = re.sub(r"[^A-Z0-9]", "", upper_text)
    if not compact:
        return ""

    car_match = re.fullmatch(r"(\d{2})([A-Z])(\d{5})", compact)
    if car_match:
        province, series, number = car_match.groups()
        return f"{province}{series}-{number[:3]}.{number[3:]}"

    motorcycle_match = re.fullmatch(r"(\d{2})([A-Z]\d)(\d{5})", compact)
    if motorcycle_match:
        province, series, number = motorcycle_match.groups()
        return f"{province}-{series} {number[:3]}.{number[3:]}"

    return compact


class PlateTextReader:
    """Đọc ký tự biển số và chuẩn hoá về chữ-số viết hoa."""

    def __init__(self) -> None:
        # Khởi tạo một lần; EasyOCR sẽ dùng lại model cho cả video.
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, plate_image: np.ndarray) -> OCRResult:
        if plate_image.size == 0:
            return OCRResult()

        grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(grayscale, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enlarged)
        filtered = cv2.bilateralFilter(enhanced, 7, 50, 50)

        first_result = self._read_candidate(filtered)
        if first_result.text and first_result.confidence >= 0.35:
            return first_result

        _, thresholded = cv2.threshold(
            filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_result = self._read_candidate(thresholded)
        return max(
            (first_result, threshold_result),
            key=lambda result: (bool(result.text), result.confidence),
        )

    def _read_candidate(self, image: np.ndarray) -> OCRResult:
        detections = self._reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=ALLOWED_CHARACTERS,
        )
        if not detections:
            return OCRResult()

        parts: list[str] = []
        weighted_confidence = 0.0
        total_weight = 0
        for _, detected_text, confidence in detections:
            text = str(detected_text).strip().upper()
            if not text:
                continue
            parts.append(text)
            weight = max(1, len(re.sub(r"[^A-Z0-9]", "", text)))
            weighted_confidence += float(confidence) * weight
            total_weight += weight

        if not parts or total_weight == 0:
            return OCRResult()

        raw_text = " ".join(parts)
        return OCRResult(
            raw_text=raw_text,
            text=normalize_plate_text(raw_text),
            confidence=max(0.0, min(1.0, weighted_confidence / total_weight)),
        )
