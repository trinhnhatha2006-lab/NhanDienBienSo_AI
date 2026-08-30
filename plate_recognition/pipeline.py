from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import PlateDetector
from .ocr import PlateTextReader
from .types import Recognition


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


class RecognitionPipeline:
    """Xử lý phát hiện, OCR và hiển thị kết quả."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        enable_ocr: bool = True,
    ) -> None:
        self.detector = PlateDetector(model_path, confidence)
        self.reader = PlateTextReader() if enable_ocr else None
        self._seen_plates: set[str] = set()
        self._scan_statuses: dict[str, str] = {}

    def process(
        self,
        source: str | Path,
        preview: bool = False,
        ocr_interval: int = 10,
    ) -> dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {source_path}")

        suffix = source_path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return self.process_video(
                source_path,
                preview=preview,
                ocr_interval=ocr_interval,
            )
        if suffix in IMAGE_EXTENSIONS:
            return self.process_image(
                source_path,
                preview=preview,
            )
        raise ValueError(
            f"Định dạng '{suffix or '(không có đuôi)'}' chưa được hỗ trợ. "
            "Hãy dùng ảnh JPG/PNG/WEBP/BMP hoặc video MP4/AVI/MOV/MKV/WMV/M4V."
        )

    def process_image(
        self,
        source: Path,
        preview: bool = False,
    ) -> dict[str, Any]:
        self._start_scan_session()
        frame = self._read_image(source)
        if frame is None:
            raise ValueError(f"Không thể đọc ảnh: {source}")

        visual, recognitions, rows = self._recognize_frame(
            frame,
            run_ocr=True,
        )
        self._print_recognitions("Ảnh", recognitions)
        report = self._base_report(str(source), "image", ocr_interval=1) | {
            "frames_processed": 1,
            "stopped_early": False,
            "detections_total": len(rows),
            "detections": rows,
            "recognized_texts": sorted(
                {item.text for item in recognitions if item.text}
            ),
            "new_plates": sorted(
                {item.text for item in recognitions if item.scan_status == "new"}
            ),
            "duplicate_plates": sorted(
                {
                    item.text
                    for item in recognitions
                    if item.scan_status == "duplicate"
                }
            ),
        }
        if preview:
            self._preview_image(visual)
        return report

    def process_video(
        self,
        source: Path,
        preview: bool = False,
        ocr_interval: int = 10,
    ) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Không thể mở video: {source}")
        return self._process_capture(
            capture=capture,
            source_label=str(source),
            kind="video",
            preview=preview,
            ocr_interval=ocr_interval,
            camera_index=None,
        )

    def process_webcam(
        self,
        camera_index: int = 0,
        preview: bool = True,
        ocr_interval: int = 10,
    ) -> dict[str, Any]:
        if camera_index < 0:
            raise ValueError("camera-index phải lớn hơn hoặc bằng 0.")
        capture = self._open_webcam(camera_index)
        return self._process_capture(
            capture=capture,
            source_label=f"camera:{camera_index}",
            kind="webcam",
            preview=preview,
            ocr_interval=ocr_interval,
            camera_index=camera_index,
        )

    def _process_capture(
        self,
        capture: cv2.VideoCapture,
        source_label: str,
        kind: str,
        preview: bool,
        ocr_interval: int,
        camera_index: int | None,
    ) -> dict[str, Any]:
        if ocr_interval < 1:
            capture.release()
            raise ValueError("ocr-interval phải lớn hơn hoặc bằng 1.")

        self._start_scan_session()
        frame_index = 0
        detections_total = 0
        latest_rows: list[dict[str, Any]] = []
        cached_ocr: list[Recognition] = []
        recognized_texts: set[str] = set()
        new_plates: set[str] = set()
        duplicate_plates: set[str] = set()
        printed_scan_states: set[tuple[str, str]] = set()
        stopped_early = False
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if frame_index == 0:
                        raise ValueError(
                            "Đã mở nguồn nhưng không đọc được khung hình đầu tiên."
                        )
                    break

                frame_index += 1
                is_interval_frame = (frame_index - 1) % ocr_interval == 0
                run_ocr = bool(self.reader is not None and is_interval_frame)
                visual, recognitions, rows = self._recognize_frame(
                    frame,
                    run_ocr=run_ocr,
                    cached_ocr=cached_ocr,
                )
                if self.reader is not None:
                    cached_ocr = recognitions
                self._refresh_scan_session(recognitions)
                recognized_texts.update(
                    item.text for item in recognitions if item.text
                )
                new_plates.update(
                    item.text
                    for item in recognitions
                    if item.scan_status == "new"
                )
                duplicate_plates.update(
                    item.text
                    for item in recognitions
                    if item.scan_status == "duplicate"
                )
                if is_interval_frame:
                    self._print_recognitions(
                        f"Frame {frame_index}",
                        recognitions,
                        printed_scan_states,
                    )

                latest_rows = [{"frame": frame_index} | row for row in rows]
                detections_total += len(latest_rows)
                if preview and self._preview_stream(visual, kind):
                    stopped_early = True
                    break
        finally:
            capture.release()
            if preview:
                self._close_windows()

        report = self._base_report(source_label, kind, ocr_interval) | {
            "frames_processed": frame_index,
            "stopped_early": stopped_early,
            "detections_total": detections_total,
            "detections": latest_rows,
            "recognized_texts": sorted(recognized_texts),
            "new_plates": sorted(new_plates),
            "duplicate_plates": sorted(duplicate_plates),
        }
        if camera_index is not None:
            report["camera_index"] = camera_index
        return report

    def _recognize_frame(
        self,
        frame: np.ndarray,
        run_ocr: bool,
        cached_ocr: list[Recognition] | None = None,
    ) -> tuple[np.ndarray, list[Recognition], list[dict[str, Any]]]:
        recognitions = self.detector.find(frame)
        if self.reader is not None:
            if run_ocr:
                for recognition in recognitions:
                    recognition.apply_ocr(self.reader.read(recognition.crop))
                if cached_ocr:
                    self._stabilize_stream_ocr(recognitions, cached_ocr)
                self._mark_scan_statuses(recognitions)
            elif cached_ocr:
                self._reuse_cached_ocr(recognitions, cached_ocr)

        rows = [recognition.as_dict() for recognition in recognitions]
        return self._draw(frame, recognitions), recognitions, rows

    def _start_scan_session(self) -> None:
        self._scan_statuses = {}

    def _refresh_scan_session(self, recognitions: list[Recognition]) -> None:
        active_texts = {item.text for item in recognitions if item.text}
        self._scan_statuses = {
            text: status
            for text, status in self._scan_statuses.items()
            if text in active_texts
        }

    def _mark_scan_statuses(self, recognitions: list[Recognition]) -> None:
        for recognition in recognitions:
            if not recognition.text or recognition.scan_status is not None:
                continue
            text = " ".join(recognition.text.upper().split())
            recognition.text = text
            if text in self._scan_statuses:
                recognition.scan_status = self._scan_statuses[text]
            elif text in self._seen_plates:
                recognition.scan_status = "duplicate"
            else:
                recognition.scan_status = "new"
                self._seen_plates.add(text)
            self._scan_statuses[text] = recognition.scan_status

    def _stabilize_stream_ocr(
        self,
        recognitions: list[Recognition],
        cached_ocr: list[Recognition],
        minimum_iou: float = 0.5,
    ) -> None:
        unused_cached = set(range(len(cached_ocr)))
        for recognition in recognitions:
            candidates = [
                (recognition.box.iou(cached_ocr[index].box), index)
                for index in unused_cached
            ]
            if not candidates:
                continue
            best_iou, best_index = max(candidates)
            previous = cached_ocr[best_index]
            if best_iou < minimum_iou:
                continue

            unused_cached.remove(best_index)
            if self._crops_are_different(recognition.crop, previous.crop):
                continue
            if not previous.text:
                continue
            if recognition.text == previous.text:
                recognition.scan_status = previous.scan_status
                recognition._ocr_candidate_text = ""
                recognition._ocr_candidate_count = 0
                continue

            candidate_text = recognition.text
            candidate_count = (
                previous._ocr_candidate_count + 1
                if candidate_text == previous._ocr_candidate_text
                else 1
            )
            if candidate_text and candidate_count >= 2:
                recognition._ocr_candidate_text = ""
                recognition._ocr_candidate_count = 0
                continue

            recognition.reuse_ocr_from(previous)
            recognition._ocr_candidate_text = candidate_text
            recognition._ocr_candidate_count = candidate_count

    @staticmethod
    def _crops_are_different(
        current: np.ndarray,
        previous: np.ndarray,
        difference_threshold: float = 0.4,
    ) -> bool:
        if current.size == 0 or previous.size == 0:
            return True
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        size = (96, 48)
        current_gray = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
        previous_gray = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
        difference = np.mean(
            np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))
        ) / 255.0
        return bool(difference > difference_threshold)

    @staticmethod
    def _reuse_cached_ocr(
        recognitions: list[Recognition],
        cached_ocr: list[Recognition],
        minimum_iou: float = 0.5,
    ) -> None:
        unused_cached = set(range(len(cached_ocr)))
        for recognition in recognitions:
            candidates = [
                (recognition.box.iou(cached_ocr[index].box), index)
                for index in unused_cached
            ]
            if not candidates:
                continue
            best_iou, best_index = max(candidates)
            previous = cached_ocr[best_index]
            if (
                best_iou >= minimum_iou
                and not RecognitionPipeline._crops_are_different(
                    recognition.crop, previous.crop
                )
            ):
                recognition.reuse_ocr_from(previous)
                unused_cached.remove(best_index)

    def _base_report(
        self, source: str, kind: str, ocr_interval: int
    ) -> dict[str, Any]:
        return {
            "source": source,
            "kind": kind,
            "model": {
                "path": str(self.detector.model_path),
                "sha256": self.detector.model_sha256,
                "class_id": self.detector.class_id,
                "class_name": self.detector.class_names[self.detector.class_id],
            },
            "confidence_threshold": self.detector.confidence,
            "ocr_enabled": self.reader is not None,
            "ocr_interval": ocr_interval,
        }

    @staticmethod
    def _draw(frame: np.ndarray, recognitions: list[Recognition]) -> np.ndarray:
        annotated = frame.copy()
        for recognition in recognitions:
            box = recognition.box
            cv2.rectangle(
                annotated,
                (box.left, box.top),
                (box.right, box.bottom),
                (0, 255, 0),
                2,
            )
            plate_text = recognition.text or "license_plate"
            status = {
                "new": "MOI",
                "duplicate": "DA QUET",
            }.get(recognition.scan_status)
            label = f"{plate_text} | {box.confidence:.0%}"
            if status:
                label += f" | {status}"
            baseline_y = max(24, box.top - 8)
            cv2.putText(
                annotated,
                label,
                (box.left, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        return annotated

    @staticmethod
    def _print_recognitions(
        source_label: str,
        recognitions: list[Recognition],
        printed_scan_states: set[tuple[str, str]] | None = None,
    ) -> None:
        if not recognitions:
            print(f"{source_label}: không phát hiện biển số")
            return

        for index, recognition in enumerate(recognitions, start=1):
            text = recognition.text or "OCR chưa đọc được"
            if recognition.text and printed_scan_states is not None:
                message_key = (recognition.text, recognition.scan_status or "")
                if message_key in printed_scan_states:
                    continue
                printed_scan_states.add(message_key)
            ocr_confidence = (
                f", OCR {recognition.ocr_confidence:.0%}"
                if recognition.ocr_confidence is not None
                else ""
            )
            scan_status = {
                "new": ", lượt quét mới",
                "duplicate": ", đã quét trong phiên này",
            }.get(recognition.scan_status, "")
            print(
                f"{source_label} - biển {index}: {text} "
                f"(YOLO {recognition.box.confidence:.0%}{ocr_confidence}"
                f"{scan_status})"
            )

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _open_webcam(camera_index: int) -> cv2.VideoCapture:
        capture: cv2.VideoCapture | None = None
        if os.name == "nt":
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise OSError(
                f"Không thể mở camera {camera_index}. Hãy kiểm tra quyền Camera của "
                "Windows, camera-index và ứng dụng khác đang sử dụng camera."
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @staticmethod
    def _preview_image(frame: np.ndarray) -> None:
        title = "Nhan dien bien so"
        try:
            cv2.imshow(title, frame)
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key == 27:
                    break
                try:
                    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV hiện không hỗ trợ cửa sổ xem trực tiếp. Hãy cài lại "
                "opencv-python (bản GUI) theo README."
            ) from exc
        finally:
            RecognitionPipeline._close_windows()

    @staticmethod
    def _preview_stream(frame: np.ndarray, kind: str) -> bool:
        title = "Nhan dien bien so"
        try:
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return True
            try:
                return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                return True
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV hiện không hỗ trợ cửa sổ xem trực tiếp. Hãy cài lại "
                "opencv-python (bản GUI) theo README."
            ) from exc
        return False

    @staticmethod
    def _close_windows() -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
