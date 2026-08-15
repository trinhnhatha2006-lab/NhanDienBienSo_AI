from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Thư mục chứa file main.py.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best.pt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nhận diện biển số xe từ ảnh hoặc video."
    )

    parser.add_argument(
        "source",
        nargs="?",
        help="Đường dẫn file ảnh hoặc video. Có thể nhập sau khi chạy.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Đường dẫn model YOLO (.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Ngưỡng tin cậy YOLO, mặc định là 0.5.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục lưu kết quả.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Hiển thị kết quả trực tiếp trong cửa sổ OpenCV.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Chỉ phát hiện biển số, không đọc ký tự.",
    )
    parser.add_argument(
        "--no-save-plates",
        action="store_true",
        help="Không lưu ảnh crop biển số.",
    )
    return parser


def ask_for_source() -> str:
    """Hỏi người dùng và kiểm tra đường dẫn ảnh/video."""
    while True:
        raw_path = input(
            "\nNhập đường dẫn đầy đủ đến ảnh hoặc video "
            "(gõ q để thoát):\n> "
        ).strip().strip('"')

        if raw_path.lower() in {"q", "quit", "exit"}:
            raise SystemExit("Đã thoát chương trình.")

        source_path = Path(raw_path).expanduser()

        if source_path.is_file():
            return str(source_path)

        print("\nKhông tìm thấy file:")
        print(source_path)
        print("Hãy kiểm tra lại tên file hoặc dùng Copy as path rồi dán vào.\n")


def main() -> int:
    args = build_parser().parse_args()

    # Khi bấm Run main.py mà không truyền lệnh:
    # chương trình hỏi đường dẫn và tự bật cửa sổ kết quả.
    if not args.source:
        args.source = ask_for_source()
        args.preview = True

        # Video có nhiều frame, không lưu vô số ảnh crop biển số.
        if Path(args.source).suffix.lower() in VIDEO_EXTENSIONS:
            args.no_save_plates = True

    # Import ở đây để lệnh --help vẫn hoạt động khi chưa cài thư viện AI.
    from plate_recognition import RecognitionPipeline

    pipeline = RecognitionPipeline(
        model_path=Path(args.model),
        confidence=args.confidence,
        output_root=args.output_dir,
        enable_ocr=not args.no_ocr,
        save_plates=not args.no_save_plates,
    )

    report = pipeline.process(args.source, preview=args.preview)

    print("\n========== KẾT QUẢ ==========")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nĐã hoàn thành. Kết quả nằm tại:\n{report['run_dir']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())