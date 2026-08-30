from __future__ import annotations

import unittest

from plate_recognition.ocr import normalize_plate_text


class NormalizePlateTextTests(unittest.TestCase):
    def test_formats_car_plate(self) -> None:
        self.assertEqual(normalize_plate_text("51A192.22"), "51A-192.22")
        self.assertEqual(normalize_plate_text("51a-192 22"), "51A-192.22")

    def test_formats_motorcycle_plate(self) -> None:
        self.assertEqual(normalize_plate_text("59-S3633.39"), "59-S3 633.39")

    def test_formats_old_four_digit_motorcycle_plate_when_structure_is_clear(self) -> None:
        self.assertEqual(normalize_plate_text("54-U5 7001"), "54-U5 7001")
        self.assertEqual(normalize_plate_text("54.U5 7001"), "54-U5 7001")

    def test_only_cleans_unknown_pattern(self) -> None:
        self.assertEqual(normalize_plate_text("AB OI-12?"), "ABOI12")

    def test_empty_input(self) -> None:
        self.assertEqual(normalize_plate_text(""), "")


if __name__ == "__main__":
    unittest.main()
