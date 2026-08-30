from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VALIDATION_GROUPS = {"clip10", "clip13"}
TEST_GROUPS = {"clip4", "clip38"}
EXPECTED_COUNTS = {
    "train": (350, 598),
    "val": (73, 104),
    "test": (75, 78),
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def canonical_group(stem: str) -> str:
    """Gộp các frame cùng video nguồn, ví dụ clip4 và clip4_new."""
    normalized = stem.lower().replace("_new", "")
    group = re.sub(r"[_-]?\d+(?:\.\d+)?$", "", normalized)
    return group or "standalone_numeric"


def target_split(group: str) -> str:
    if group in TEST_GROUPS:
        return "test"
    if group in VALIDATION_GROUPS:
        return "val"
    return "train"


def validate_label(path: Path) -> int:
    box_count = 0
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: nhãn phải có đúng 5 trường")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: nhãn không phải số hợp lệ") from exc
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: class phải là 0")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{path}:{line_number}: tâm box nằm ngoài [0, 1]")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: kích thước box không hợp lệ")
        box_count += 1
    if box_count == 0:
        raise ValueError(f"{path}: file nhãn rỗng")
    return box_count


def collect_entries(dataset_root: Path) -> dict[str, list[tuple[Path, int, str]]]:
    entries: dict[str, list[tuple[Path, int, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    seen_images: set[Path] = set()

    for original_split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / original_split
        label_dir = dataset_root / "labels" / original_split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục ảnh hoặc nhãn của split {original_split}")

        image_paths = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_stems = {path.stem for path in image_paths}
        orphan_labels = sorted(
            path
            for path in label_dir.glob("*.txt")
            if path.stem not in image_stems
        )
        if orphan_labels:
            raise ValueError(f"Nhãn không có ảnh tương ứng: {orphan_labels[0]}")

        for image_path in sorted(image_paths, key=lambda item: item.name.lower()):
            if image_path in seen_images:
                raise ValueError(f"Ảnh bị lặp trong quá trình quét: {image_path}")
            seen_images.add(image_path)

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Ảnh chưa có nhãn tương ứng: {image_path}")

            group = canonical_group(image_path.stem)
            split = target_split(group)
            entries[split].append((image_path, validate_label(label_path), group))

    return entries


def relative_manifest_path(image_path: Path, splits_dir: Path) -> str:
    relative = image_path.relative_to(splits_dir.parent)
    return f"./../{relative.as_posix()}"


def write_manifests(
    entries: dict[str, list[tuple[Path, int, str]]], splits_dir: Path
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, split_entries in entries.items():
        lines = [relative_manifest_path(path, splits_dir) for path, _, _ in split_entries]
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def validate_expected_counts(entries: dict[str, list[tuple[Path, int, str]]]) -> None:
    for split, expected in EXPECTED_COUNTS.items():
        image_count = len(entries[split])
        box_count = sum(boxes for _, boxes, _ in entries[split])
        if (image_count, box_count) != expected:
            raise ValueError(
                f"Split {split} có {image_count} ảnh/{box_count} box, "
                f"khác dự kiến {expected[0]} ảnh/{expected[1]} box"
            )


def print_summary(entries: dict[str, list[tuple[Path, int, str]]]) -> None:
    for split in ("train", "val", "test"):
        split_entries = entries[split]
        groups = Counter(group for _, _, group in split_entries)
        boxes = sum(box_count for _, box_count, _ in split_entries)
        print(
            f"{split:5}: {len(split_entries):3} ảnh, {boxes:3} box, "
            f"{len(groups):2} nhóm nguồn"
        )
        print(f"       {', '.join(sorted(groups))}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra dataset và tạo split YOLO theo video/clip nguồn."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Ghi các manifest vào datasets/license_plate/splits.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "datasets" / "license_plate"
    entries = collect_entries(dataset_root)
    validate_expected_counts(entries)
    print_summary(entries)

    if args.write:
        splits_dir = dataset_root / "splits"
        write_manifests(entries, splits_dir)
        print(f"Đã ghi manifest tại: {splits_dir}")
    else:
        print("Chỉ kiểm tra; thêm --write nếu muốn tạo lại manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
