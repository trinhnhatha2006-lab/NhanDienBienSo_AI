from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OCRResult:
    """Kết quả OCR trước và sau khi chuẩn hoá."""

    raw_text: str = ""
    text: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class BoundingBox:
    """Hình chữ nhật phát hiện theo toạ độ pixel trên ảnh gốc."""

    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    def iou(self, other: "BoundingBox") -> float:
        """Tính Intersection over Union giữa hai bounding box."""
        intersection_left = max(self.left, other.left)
        intersection_top = max(self.top, other.top)
        intersection_right = min(self.right, other.right)
        intersection_bottom = min(self.bottom, other.bottom)

        intersection_width = max(0, intersection_right - intersection_left)
        intersection_height = max(0, intersection_bottom - intersection_top)
        intersection_area = intersection_width * intersection_height
        if intersection_area == 0:
            return 0.0

        self_area = (self.right - self.left) * (self.bottom - self.top)
        other_area = (other.right - other.left) * (other.bottom - other.top)
        union_area = self_area + other_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Recognition:
    """Một biển số đã phát hiện, kèm ảnh crop và kết quả OCR."""

    box: BoundingBox
    crop: np.ndarray
    raw_text: str = ""
    text: str = ""
    ocr_confidence: float | None = None
    ocr_reused: bool = False
    scan_status: str | None = None
    _ocr_candidate_text: str = ""
    _ocr_candidate_count: int = 0

    def apply_ocr(self, result: OCRResult, reused: bool = False) -> None:
        self.raw_text = result.raw_text
        self.text = result.text
        self.ocr_confidence = result.confidence
        self.ocr_reused = reused

    def reuse_ocr_from(self, other: "Recognition") -> None:
        self.raw_text = other.raw_text
        self.text = other.text
        self.ocr_confidence = other.ocr_confidence
        self.ocr_reused = bool(other.raw_text or other.text)
        self.scan_status = other.scan_status
        self._ocr_candidate_text = other._ocr_candidate_text
        self._ocr_candidate_count = other._ocr_candidate_count

    def as_dict(self) -> dict[str, Any]:
        return self.box.as_dict() | {
            "raw_text": self.raw_text,
            "text": self.text,
            "ocr_confidence": (
                round(self.ocr_confidence, 4)
                if self.ocr_confidence is not None
                else None
            ),
            "ocr_reused": self.ocr_reused,
            "scan_status": self.scan_status,
        }
