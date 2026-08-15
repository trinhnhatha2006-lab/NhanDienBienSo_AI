from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import PlateDetector
from .ocr import PlateTextReader
from .storage import RunStorage
from .types import Recognition


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


class RecognitionPipeline:
    """Điều phối phát hiện, OCR, vẽ kết quả và lưu artefact."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        output_root: str | Path = "runs",
        enable_ocr: bool = True,
        save_plates: bool = True,
    ) -> None:
        self.detector = PlateDetector(model_path, confidence)
        self.reader = PlateTextReader() if enable_ocr else None
        self.output_root = Path(output_root)
        self.save_plates = save_plates

    def process(self, source: str | Path, preview: bool = False) -> dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {source_path}")
        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            return self.process_video(source_path, preview)
        return self.process_image(source_path, preview)

    def process_image(self, source: Path, preview: bool = False) -> dict[str, Any]:
        frame = self._read_image(source)
        if frame is None:
            raise ValueError(f"Không thể đọc ảnh: {source}")

        storage = RunStorage(self.output_root)
        visual, recognitions, rows = self._recognize_frame(frame, storage)
        output = storage.save_image(visual)
        report = {
            "source": str(source),
            "kind": "image",
            "output": str(output),
            "detections": rows,
        }
        storage.save_report(report)
        if preview:
            self._preview_image(visual)
        return report | {"run_dir": str(storage.run_dir)}

    def process_video(self, source: Path, preview: bool = False) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Không thể mở video: {source}")

        storage = RunStorage(self.output_root)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = storage.run_dir / "annotated_video.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise OSError(f"Không thể tạo video đầu ra: {output}")

        frame_index = 0
        detection_rows: list[dict[str, Any]] = []
        stopped_early = False
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                visual, _, rows = self._recognize_frame(frame, storage)
                detection_rows.extend({"frame": frame_index} | row for row in rows)
                writer.write(visual)
                if preview and self._preview_video(visual):
                    stopped_early = True
                    break
        finally:
            capture.release()
            writer.release()
            if preview:
                cv2.destroyAllWindows()

        report = {
            "source": str(source),
            "kind": "video",
            "output": str(output),
            "frames_processed": frame_index,
            "stopped_early": stopped_early,
            "detections": detection_rows,
        }
        storage.save_report(report)
        return report | {"run_dir": str(storage.run_dir)}

    def _recognize_frame(
        self, frame: np.ndarray, storage: RunStorage
    ) -> tuple[np.ndarray, list[Recognition], list[dict[str, Any]]]:
        recognitions = self.detector.find(frame)
        rows: list[dict[str, Any]] = []
        for recognition in recognitions:
            if self.reader is not None:
                recognition.text = self.reader.read(recognition.crop)
            crop_file = storage.save_plate(recognition.crop) if self.save_plates else None
            rows.append(recognition.as_dict(str(crop_file) if crop_file else None))
        return self._draw(frame, recognitions), recognitions, rows

    @staticmethod
    def _draw(frame: np.ndarray, recognitions: list[Recognition]) -> np.ndarray:
        annotated = frame.copy()
        for recognition in recognitions:
            box = recognition.box
            cv2.rectangle(annotated, (box.left, box.top), (box.right, box.bottom), (0, 255, 0), 2)
            label = recognition.text or f"plate {box.confidence:.0%}"
            baseline_y = max(24, box.top - 8)
            cv2.putText(annotated, label, (box.left, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return annotated

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _preview_image(frame: np.ndarray) -> None:
        cv2.imshow("Nhan dien bien so", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _preview_video(frame: np.ndarray) -> bool:
        cv2.imshow("Nhan dien bien so - Nhan q de dung", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")
