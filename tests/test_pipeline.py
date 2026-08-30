from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from plate_recognition.pipeline import RecognitionPipeline
from plate_recognition.types import BoundingBox, OCRResult, Recognition


class EmptyDetector:
    model_path = Path("fake.pt")
    model_sha256 = "0" * 64
    class_id = 0
    class_names = {0: "license_plate"}
    confidence = 0.5

    @staticmethod
    def find(_frame: np.ndarray) -> list[object]:
        return []


class SingleDetector(EmptyDetector):
    @staticmethod
    def find(frame: np.ndarray) -> list[Recognition]:
        return [
            Recognition(
                box=BoundingBox(1, 1, 10, 10, 0.9),
                crop=frame[1:10, 1:10].copy(),
            )
        ]


class FixedReader:
    @staticmethod
    def read(_crop: np.ndarray) -> OCRResult:
        return OCRResult(
            raw_text="54-U5 7001",
            text="54-U5 7001",
            confidence=0.97,
        )


class ChangingReader:
    def __init__(self) -> None:
        self.results = iter(
            [
                OCRResult("59-S3 633.39", "59-S3 633.39", 0.71),
                OCRResult("595363339", "595363339", 0.83),
                OCRResult("S35963339", "S35963339", 0.83),
            ]
        )

    def read(self, _crop: np.ndarray) -> OCRResult:
        return next(self.results)


class FakeCapture:
    def __init__(self, frame_count: int = 1) -> None:
        self.frames = [
            np.zeros((30, 40, 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        self.released = False

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    @staticmethod
    def get(_property: int) -> float:
        return 25.0

    def release(self) -> None:
        self.released = True


class LeavingDetector(EmptyDetector):
    def __init__(self) -> None:
        self.frame_number = 0

    def find(self, frame: np.ndarray) -> list[Recognition]:
        self.frame_number += 1
        if self.frame_number == 2:
            return []
        return SingleDetector.find(frame)


class DirectModeTests(unittest.TestCase):
    @staticmethod
    def pipeline(detector=EmptyDetector()):
        pipeline = RecognitionPipeline.__new__(RecognitionPipeline)
        pipeline.detector = detector
        pipeline.reader = None
        pipeline._seen_plates = set()
        pipeline._scan_statuses = {}
        return pipeline

    def test_direct_image_mode_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "recognition"
            pipeline = self.pipeline()

            frame = np.zeros((50, 80, 3), dtype=np.uint8)
            with patch.object(RecognitionPipeline, "_read_image", return_value=frame):
                report = pipeline.process_image(Path("no_plate.jpg"), preview=False)

            self.assertFalse(output_root.exists())
            self.assertNotIn("run_dir", report)
            self.assertNotIn("output", report)
            self.assertNotIn("report_file", report)
            self.assertEqual(report["frames_processed"], 1)
            self.assertFalse(report["stopped_early"])
            self.assertEqual(report["detections"], [])

    def test_same_image_is_duplicate_on_second_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self.pipeline(SingleDetector())
            pipeline.reader = FixedReader()
            frame = np.zeros((50, 80, 3), dtype=np.uint8)

            with (
                patch.object(RecognitionPipeline, "_read_image", return_value=frame),
                patch("builtins.print"),
            ):
                first = pipeline.process_image(Path("xe_01.jpg"), preview=False)
                second = pipeline.process_image(Path("xe_01.jpg"), preview=False)

            self.assertEqual(first["new_plates"], ["54-U5 7001"])
            self.assertEqual(first["duplicate_plates"], [])
            self.assertEqual(first["detections"][0]["scan_status"], "new")
            self.assertEqual(second["new_plates"], [])
            self.assertEqual(second["duplicate_plates"], ["54-U5 7001"])
            self.assertEqual(second["detections"][0]["scan_status"], "duplicate")
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_new_program_session_starts_with_empty_history(self) -> None:
        first_pipeline = self.pipeline(SingleDetector())
        second_pipeline = self.pipeline(SingleDetector())
        first_pipeline.reader = FixedReader()
        second_pipeline.reader = FixedReader()
        frame = np.zeros((50, 80, 3), dtype=np.uint8)

        with (
            patch.object(RecognitionPipeline, "_read_image", return_value=frame),
            patch("builtins.print"),
        ):
            first = first_pipeline.process_image(Path("xe_01.jpg"), preview=False)
            restarted = second_pipeline.process_image(
                Path("xe_01.jpg"), preview=False
            )

        self.assertEqual(first["new_plates"], ["54-U5 7001"])
        self.assertEqual(restarted["new_plates"], ["54-U5 7001"])
        self.assertEqual(restarted["duplicate_plates"], [])

    def test_repeated_video_frames_count_as_one_scan(self) -> None:
        pipeline = self.pipeline(SingleDetector())
        pipeline.reader = FixedReader()

        with patch("builtins.print"):
            report = pipeline._process_capture(
                capture=FakeCapture(frame_count=3),
                source_label="video.mp4",
                kind="video",
                preview=False,
                ocr_interval=1,
                camera_index=None,
            )

        self.assertEqual(report["new_plates"], ["54-U5 7001"])
        self.assertEqual(report["duplicate_plates"], [])
        self.assertEqual(pipeline._seen_plates, {"54-U5 7001"})

    def test_ocr_variations_on_same_tracked_plate_do_not_create_duplicates(self) -> None:
        pipeline = self.pipeline(SingleDetector())
        pipeline.reader = ChangingReader()

        with patch("builtins.print"):
            report = pipeline._process_capture(
                capture=FakeCapture(frame_count=3),
                source_label="video.mp4",
                kind="video",
                preview=False,
                ocr_interval=1,
                camera_index=None,
            )

        self.assertEqual(report["recognized_texts"], ["59-S3 633.39"])
        self.assertEqual(report["new_plates"], ["59-S3 633.39"])
        self.assertEqual(report["duplicate_plates"], [])

    def test_plate_returning_after_leaving_is_duplicate(self) -> None:
        pipeline = self.pipeline(LeavingDetector())
        pipeline.reader = FixedReader()

        with patch("builtins.print"):
            report = pipeline._process_capture(
                capture=FakeCapture(frame_count=3),
                source_label="video.mp4",
                kind="video",
                preview=False,
                ocr_interval=1,
                camera_index=None,
            )

        self.assertEqual(report["new_plates"], ["54-U5 7001"])
        self.assertEqual(report["duplicate_plates"], ["54-U5 7001"])

    def test_stable_new_text_replaces_cached_text_after_confirmation(self) -> None:
        pipeline = self.pipeline()
        pipeline._start_scan_session()
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        previous = Recognition(BoundingBox(0, 0, 100, 100, 0.9), crop)
        previous.apply_ocr(OCRResult("51A19222", "51A-192.22", 0.95))

        first = Recognition(BoundingBox(0, 0, 100, 100, 0.9), crop)
        first.apply_ocr(OCRResult("59S363339", "59-S3 633.39", 0.99))
        pipeline._stabilize_stream_ocr([first], [previous])
        self.assertEqual(first.text, "51A-192.22")

        second = Recognition(BoundingBox(0, 0, 100, 100, 0.9), crop)
        second.apply_ocr(OCRResult("59S363339", "59-S3 633.39", 0.99))
        pipeline._stabilize_stream_ocr([second], [first])
        self.assertEqual(second.text, "59-S3 633.39")

    def test_stabilization_matches_each_cached_box_once(self) -> None:
        pipeline = self.pipeline()
        pipeline._start_scan_session()
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        cached_first = Recognition(BoundingBox(0, 0, 100, 100, 0.9), crop)
        cached_first.apply_ocr(OCRResult("A", "A", 0.9))
        cached_second = Recognition(BoundingBox(20, 0, 120, 100, 0.9), crop)
        cached_second.apply_ocr(OCRResult("B", "B", 0.9))

        current_first = Recognition(BoundingBox(0, 0, 100, 100, 0.9), crop)
        current_first.apply_ocr(OCRResult("A", "A", 0.9))
        current_second = Recognition(BoundingBox(5, 0, 105, 100, 0.9), crop)
        current_second.apply_ocr(OCRResult("X", "X", 0.9))

        pipeline._stabilize_stream_ocr(
            [current_first, current_second], [cached_first, cached_second]
        )

        self.assertEqual(current_first.text, "A")
        self.assertEqual(current_second.text, "B")

    def test_report_contains_scan_status_without_writing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self.pipeline(SingleDetector())
            pipeline.reader = FixedReader()
            frame = np.zeros((50, 80, 3), dtype=np.uint8)

            with (
                patch.object(RecognitionPipeline, "_read_image", return_value=frame),
                patch("builtins.print"),
            ):
                report = pipeline.process_image(Path("xe_01.jpg"), preview=False)

            self.assertEqual(report["new_plates"], ["54-U5 7001"])
            self.assertEqual(report["detections"][0]["scan_status"], "new")
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_direct_stream_keeps_only_latest_frame_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "recognition"
            pipeline = self.pipeline(SingleDetector())
            capture = FakeCapture(frame_count=3)

            report = pipeline._process_capture(
                capture=capture,
                source_label="camera:0",
                kind="webcam",
                preview=False,
                ocr_interval=10,
                camera_index=0,
            )

            self.assertTrue(capture.released)
            self.assertFalse(output_root.exists())
            self.assertEqual(report["detections_total"], 3)
            self.assertEqual(len(report["detections"]), 1)

    def test_capture_is_released_when_frame_processing_fails(self) -> None:
        pipeline = self.pipeline(SingleDetector())
        capture = FakeCapture(frame_count=1)

        with patch.object(
            pipeline,
            "_recognize_frame",
            side_effect=RuntimeError("frame processing failed"),
        ):
            with self.assertRaises(RuntimeError):
                pipeline._process_capture(
                    capture=capture,
                    source_label="camera:0",
                    kind="webcam",
                    preview=False,
                    ocr_interval=10,
                    camera_index=0,
                )

        self.assertTrue(capture.released)

    def test_image_preview_closes_windows_when_wait_fails(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        with (
            patch("plate_recognition.pipeline.cv2.imshow"),
            patch(
                "plate_recognition.pipeline.cv2.waitKey",
                side_effect=RuntimeError("interrupted"),
            ),
            patch.object(RecognitionPipeline, "_close_windows") as close_windows,
        ):
            with self.assertRaises(RuntimeError):
                RecognitionPipeline._preview_image(frame)

        close_windows.assert_called_once_with()

    def test_image_preview_stops_when_window_is_closed(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        with (
            patch("plate_recognition.pipeline.cv2.imshow"),
            patch("plate_recognition.pipeline.cv2.waitKey", return_value=-1),
            patch(
                "plate_recognition.pipeline.cv2.getWindowProperty",
                return_value=0,
            ),
            patch.object(RecognitionPipeline, "_close_windows") as close_windows,
        ):
            RecognitionPipeline._preview_image(frame)

        close_windows.assert_called_once_with()

    def test_stream_preview_stops_on_escape(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        with (
            patch("plate_recognition.pipeline.cv2.imshow"),
            patch("plate_recognition.pipeline.cv2.waitKey", return_value=27),
        ):
            self.assertTrue(RecognitionPipeline._preview_stream(frame, "video"))

    def test_stream_preview_stops_when_window_is_closed(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        with (
            patch("plate_recognition.pipeline.cv2.imshow"),
            patch("plate_recognition.pipeline.cv2.waitKey", return_value=-1),
            patch(
                "plate_recognition.pipeline.cv2.getWindowProperty",
                return_value=0,
            ),
        ):
            self.assertTrue(RecognitionPipeline._preview_stream(frame, "webcam"))


if __name__ == "__main__":
    unittest.main()
