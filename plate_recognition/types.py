from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Hình chữ nhật phát hiện theo toạ độ pixel trên ảnh gốc."""

    left: int
    top: int
    right: int
    bottom: int
    confidence: float

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
    text: str = ""

    def as_dict(self, crop_file: str | None = None) -> dict[str, Any]:
        payload = self.box.as_dict() | {"text": self.text}
        if crop_file:
            payload["crop_file"] = crop_file
        return payload
