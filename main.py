from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
VENV_PYTHON = BASE_DIR / ".venv-train" / "Scripts" / "python.exe"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
DEFAULT_CONFIDENCE = 0.85
OCR_INTERVAL = 10


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings(
    "ignore",
    message=r"torch\.quantize_per_tensor.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"'pin_memory' argument is set as true.*",
    category=UserWarning,
)


def ensure_project_python() -> None:
    if not sys.platform.startswith("win"):
        return

    try:
        using_venv = Path(sys.executable).samefile(VENV_PYTHON)
    except OSError:
        using_venv = False

    if using_venv:
        return
    if not VENV_PYTHON.is_file():
        print("Không tìm thấy môi trường .venv-train.", file=sys.stderr)
        print(
            "Hãy chạy: py -3.11 -m venv .venv-train",
            file=sys.stderr,
        )
        raise SystemExit(2)

    completed = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(BASE_DIR),
        check=False,
    )
    raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nhận diện biển số xe từ ảnh, video hoặc webcam."
    )
    parser.add_argument("source", nargs="?", help="Đường dẫn ảnh hoặc video.")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Nhận diện trực tiếp bằng webcam.",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Đường dẫn model YOLO.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Ngưỡng phát hiện, mặc định 0.85.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Số thứ tự webcam, mặc định 0.",
    )
    return parser


def clean_input_path(source: str | Path) -> str:
    value = str(source).strip()
    if value.startswith("&"):
        value = value[1:].lstrip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        value = value[1:-1].strip()
    if not value:
        raise ValueError("Bạn chưa nhập đường dẫn.")
    return value


def resolve_source_path(source: str | Path) -> Path:
    value = clean_input_path(source)
    path = Path(value).expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [BASE_DIR / path, BASE_DIR / "BienSoXe" / path, Path.cwd() / path]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError("Chương trình chỉ nhận file ảnh hoặc video.")
            return resolved

    raise FileNotFoundError(f"Không tìm thấy file: {candidates[0].resolve()}")


def ask_for_source() -> tuple[str | None, bool]:
    while True:
        print("\nNhập đường dẫn ảnh/video hoặc gõ cam để mở webcam")
        value = input("> ").strip()
        if value.lower() in {"cam", "camera", "webcam"}:
            return None, True
        try:
            return str(resolve_source_path(value)), False
        except (FileNotFoundError, ValueError) as exc:
            print(exc)


def print_report(report: dict[str, object]) -> None:
    texts = report.get("recognized_texts")
    if not isinstance(texts, list):
        texts = []

    print("\n========== KẾT QUẢ ==========")
    print(f"Loại nguồn: {report.get('kind')}")
    print(f"Số frame: {report.get('frames_processed', 0)}")
    print(f"Số lượt phát hiện: {report.get('detections_total', 0)}")
    print("Biển số: " + (", ".join(texts) if texts else "chưa đọc được"))

    new_plates = report.get("new_plates")
    duplicate_plates = report.get("duplicate_plates")
    if isinstance(new_plates, list) and new_plates:
        print("Lượt quét mới: " + ", ".join(new_plates))
    if isinstance(duplicate_plates, list) and duplicate_plates:
        print("Đã quét trước đó: " + ", ".join(duplicate_plates))


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.source and args.webcam:
        parser.error("Không dùng đường dẫn cùng lúc với --webcam.")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence phải nằm trong khoảng (0, 1].")
    if args.camera_index < 0:
        parser.error("--camera-index phải lớn hơn hoặc bằng 0.")
    return args


def main() -> int:
    parser = build_parser()
    args = validate_args(parser, parser.parse_args())
    interactive = not args.source and not args.webcam

    try:
        model = Path(args.model).expanduser()
        if not model.is_absolute():
            model = BASE_DIR / model
        model = model.resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Không tìm thấy model: {model}")

        from plate_recognition import RecognitionPipeline

        pipeline = RecognitionPipeline(
            model_path=model,
            confidence=args.confidence,
        )
    except ImportError:
        print(
            "Thiếu thư viện. Hãy chạy: "
            ".\\.venv-train\\Scripts\\python.exe -m pip install "
            "-r requirements.txt",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2

    while True:
        try:
            if interactive:
                source_value, use_webcam = ask_for_source()
            else:
                source_value, use_webcam = args.source, args.webcam

            if use_webcam:
                report = pipeline.process_webcam(
                    camera_index=args.camera_index,
                    preview=True,
                    ocr_interval=OCR_INTERVAL,
                )
            else:
                report = pipeline.process(
                    resolve_source_path(source_value),
                    preview=True,
                    ocr_interval=OCR_INTERVAL,
                )
            print_report(report)
        except KeyboardInterrupt:
            print("\nĐã dừng chương trình.")
            return 130
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            print(f"Lỗi: {exc}", file=sys.stderr)
            if not interactive:
                return 2

        if not interactive:
            return 0


if __name__ == "__main__":
    ensure_project_python()
    raise SystemExit(main())
