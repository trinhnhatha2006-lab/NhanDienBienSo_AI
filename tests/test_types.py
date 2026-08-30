from __future__ import annotations

import unittest

import numpy as np

from plate_recognition.pipeline import RecognitionPipeline
from plate_recognition.types import BoundingBox, OCRResult, Recognition


def recognition(box: BoundingBox) -> Recognition:
    return Recognition(box=box, crop=np.zeros((10, 10, 3), dtype=np.uint8))


class BoundingBoxTests(unittest.TestCase):
    def test_iou(self) -> None:
        first = BoundingBox(0, 0, 100, 100, 0.9)
        second = BoundingBox(50, 50, 150, 150, 0.8)
        self.assertAlmostEqual(first.iou(second), 2500 / 17500)

    def test_reuses_ocr_for_overlapping_box(self) -> None:
        cached = recognition(BoundingBox(0, 0, 100, 100, 0.9))
        cached.apply_ocr(OCRResult("51A19222", "51A-192.22", 0.95))
        current = recognition(BoundingBox(5, 5, 105, 105, 0.88))

        RecognitionPipeline._reuse_cached_ocr([current], [cached])

        self.assertEqual(current.text, "51A-192.22")
        self.assertTrue(current.ocr_reused)

    def test_does_not_reuse_ocr_for_distant_box(self) -> None:
        cached = recognition(BoundingBox(0, 0, 100, 100, 0.9))
        cached.apply_ocr(OCRResult("51A19222", "51A-192.22", 0.95))
        current = recognition(BoundingBox(200, 200, 300, 300, 0.88))

        RecognitionPipeline._reuse_cached_ocr([current], [cached])

        self.assertEqual(current.text, "")
        self.assertFalse(current.ocr_reused)


if __name__ == "__main__":
    unittest.main()
