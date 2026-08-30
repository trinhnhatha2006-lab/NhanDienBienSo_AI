from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_THRESHOLDS = tuple(round(value / 100, 2) for value in range(25, 90, 5))


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Đánh giá detector và chọn confidence theo precision/recall/F1."
    )
    parser.add_argument("--model", required=True, help="Đường dẫn model YOLO .pt")
    parser.add_argument(
        "--manifest",
        default=str(project_root / "datasets/license_plate/splits/val.txt"),
        help="Manifest ảnh cần đánh giá.",
    )
    parser.add_argument("--output", help="File JSON lưu metric; bỏ trống để chỉ in.")
    parser.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU ghép TP.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_images(manifest: Path) -> list[Path]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {manifest}")
    images: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("./"):
            path = (manifest.parent / line[2:]).resolve()
        else:
            path = Path(line).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh trong manifest: {path}")
        images.append(path)
    if not images:
        raise ValueError(f"Manifest không có ảnh: {manifest}")
    return images


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if not image_indices:
        raise ValueError(f"Đường dẫn ảnh không chứa thư mục images: {image_path}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return image


def read_ground_truth(label_path: Path, width: int, height: int) -> np.ndarray:
    if not label_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy nhãn: {label_path}")
    boxes: list[list[float]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) != 5 or int(parts[0]) != 0:
            raise ValueError(f"Nhãn không hợp lệ: {label_path}:{line_number}")
        x_center, y_center, box_width, box_height = map(float, parts[1:])
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def pairwise_iou(predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return np.zeros((len(predictions), len(ground_truth)), dtype=np.float32)
    top_left = np.maximum(predictions[:, None, :2], ground_truth[None, :, :2])
    bottom_right = np.minimum(predictions[:, None, 2:], ground_truth[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    prediction_area = (
        (predictions[:, 2] - predictions[:, 0])
        * (predictions[:, 3] - predictions[:, 1])
    )
    ground_truth_area = (
        (ground_truth[:, 2] - ground_truth[:, 0])
        * (ground_truth[:, 3] - ground_truth[:, 1])
    )
    union = prediction_area[:, None] + ground_truth_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_counts(
    prediction_boxes: np.ndarray,
    prediction_scores: np.ndarray,
    ground_truth: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int]:
    order = np.argsort(-prediction_scores)
    prediction_boxes = prediction_boxes[order]
    ious = pairwise_iou(prediction_boxes, ground_truth)
    matched_ground_truth: set[int] = set()
    true_positives = 0

    for prediction_index in range(len(prediction_boxes)):
        candidates = [
            (float(ious[prediction_index, gt_index]), gt_index)
            for gt_index in range(len(ground_truth))
            if gt_index not in matched_ground_truth
        ]
        if not candidates:
            continue
        best_iou, best_gt_index = max(candidates)
        if best_iou >= iou_threshold:
            matched_ground_truth.add(best_gt_index)
            true_positives += 1

    false_positives = len(prediction_boxes) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def metric_row(threshold: float, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows if row["precision"] >= 0.9 and row["recall"] >= 0.9
    ]
    candidates = eligible or rows
    return max(
        candidates,
        key=lambda row: (row["f1"], row["precision"], -row["threshold"]),
    )


def evaluate(
    model_path: Path,
    manifest: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    if not 0 < iou_threshold <= 1:
        raise ValueError("--iou phải nằm trong khoảng (0, 1].")

    model = YOLO(str(model_path))
    names = model.names
    names = dict(enumerate(names)) if isinstance(names, list) else dict(names)
    class_ids = [
        class_id
        for class_id, name in names.items()
        if str(name).strip().lower() in {"license_plate", "license plate"}
    ]
    if class_ids:
        class_id = int(class_ids[0])
    elif len(names) == 1:
        class_id = int(next(iter(names)))
    else:
        raise ValueError("Model có nhiều class nhưng không có class 'license_plate'.")
    images = resolve_manifest_images(manifest)
    counts = {threshold: [0, 0, 0] for threshold in DEFAULT_THRESHOLDS}

    for image_index, image_path in enumerate(images, start=1):
        image = read_image(image_path)
        height, width = image.shape[:2]
        ground_truth = read_ground_truth(
            image_to_label_path(image_path), width=width, height=height
        )
        result = model.predict(
            image,
            conf=min(DEFAULT_THRESHOLDS),
            classes=[class_id],
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]
        if result.boxes is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

        for threshold in DEFAULT_THRESHOLDS:
            selected = scores >= threshold
            tp, fp, fn = match_counts(
                boxes[selected], scores[selected], ground_truth, iou_threshold
            )
            counts[threshold][0] += tp
            counts[threshold][1] += fp
            counts[threshold][2] += fn
        print(f"[{image_index:3}/{len(images)}] {image_path.name}")

    rows = [
        metric_row(threshold, *counts[threshold])
        for threshold in DEFAULT_THRESHOLDS
    ]
    selected = choose_threshold(rows)
    return {
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "manifest": str(manifest.resolve()),
        "images": len(images),
        "iou_threshold": iou_threshold,
        "thresholds": rows,
        "selected": selected,
    }


def main() -> int:
    args = parse_args()
    report = evaluate(
        Path(args.model).expanduser().resolve(),
        Path(args.manifest).expanduser().resolve(),
        args.iou,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Đã lưu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
