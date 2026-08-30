from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_summary(run_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"run_dir": str(run_dir)}
    args_file = run_dir / "args.yaml"
    if args_file.is_file():
        training_args = yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}
        summary["args"] = {
            key: training_args.get(key)
            for key in (
                "model",
                "data",
                "epochs",
                "patience",
                "batch",
                "imgsz",
                "device",
                "workers",
                "seed",
            )
        }

    results_file = run_dir / "results.csv"
    if results_file.is_file():
        with results_file.open(encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if rows:
            cleaned_rows = [
                {
                    key.strip(): value
                    for key, value in row.items()
                    if key is not None
                }
                for row in rows
            ]
            summary["completed_epochs"] = len(cleaned_rows)
            summary["early_stopped"] = bool(
                summary.get("args", {}).get("epochs", len(cleaned_rows))
                > len(cleaned_rows)
            )
            summary["last_epoch_metrics"] = cleaned_rows[-1]
            summary["best_epoch_metrics"] = max(
                cleaned_rows,
                key=lambda row: (
                    0.1 * float(row["metrics/mAP50(B)"])
                    + 0.9 * float(row["metrics/mAP50-95(B)"])
                ),
            )
    return summary


def relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def split_metadata(project_root: Path) -> dict[str, Any]:
    splits_dir = project_root / "datasets" / "license_plate" / "splits"
    result: dict[str, Any] = {"strategy": "grouped-by-source-clip"}
    for split in ("train", "val", "test"):
        manifest = splits_dir / f"{split}.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"Thiếu manifest dữ liệu: {manifest}")
        image_count = sum(
            1
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        result[split] = {
            "manifest": relative_or_absolute(manifest, project_root),
            "images": image_count,
        }
    return result


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sao lưu model cũ và triển khai model YOLO đã đánh giá."
    )
    parser.add_argument("--model", required=True, help="Model best.pt mới.")
    parser.add_argument(
        "--evaluation",
        required=True,
        help="JSON hiệu chỉnh confidence từ evaluate_detector.py.",
    )
    parser.add_argument(
        "--destination",
        default=str(project_root / "models/best.pt"),
        help="Model ứng dụng; mặc định models/best.pt.",
    )
    args = parser.parse_args()

    candidate = Path(args.model).expanduser().resolve()
    evaluation_path = Path(args.evaluation).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Không tìm thấy model mới: {candidate}")
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy báo cáo đánh giá: {evaluation_path}")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected = evaluation.get("selected")
    if not isinstance(selected, dict) or "threshold" not in selected:
        raise ValueError("Báo cáo đánh giá chưa có threshold được chọn.")
    default_confidence = float(selected["threshold"])
    if not 0 < default_confidence <= 1:
        raise ValueError("Threshold trong báo cáo đánh giá không hợp lệ.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_dir = destination.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.parent / "model_metadata.json"
    existing_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                existing_metadata = loaded_metadata
        except json.JSONDecodeError:
            pass
    backup_path: Path | None = None
    candidate_hash = sha256(candidate)
    evaluated_hash = str(evaluation.get("model_sha256", "")).lower()
    if not evaluated_hash:
        raise ValueError(
            "Báo cáo đánh giá thiếu model_sha256; hãy chạy lại evaluate_detector.py."
        )
    if evaluated_hash != candidate_hash.lower():
        raise ValueError(
            "Model cần triển khai không khớp model trong báo cáo đánh giá."
        )

    if destination.is_file():
        current_hash = sha256(destination)
        if current_hash != candidate_hash:
            modified_date = datetime.fromtimestamp(
                destination.stat().st_mtime
            ).strftime("%Y%m%d")
            backup_path = archive_dir / (
                f"best_legacy_{modified_date}_{current_hash[:8]}.pt"
            )
    run_dir = candidate.parent.parent
    backup_reference = (
        relative_or_absolute(backup_path, project_root)
        if backup_path
        else existing_metadata.get("backup_model")
    )
    metadata = {
        "schema_version": 1,
        "deployed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_file": destination.name,
        "model_sha256": candidate_hash,
        "source_model": relative_or_absolute(candidate, project_root),
        "data_config": "datasets/license_plate/data.yaml",
        "dataset_splits": split_metadata(project_root),
        "default_confidence": default_confidence,
        "evaluation": evaluation,
        "training": read_training_summary(run_dir),
        "backup_model": backup_reference,
    }
    metadata_payload = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"

    # Tạo model và metadata tạm trước khi thay bản đang dùng.
    temporary_destination = destination.with_suffix(".pt.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    rollback_destination = destination.with_suffix(".pt.rollback")
    shutil.copy2(candidate, temporary_destination)
    temporary_metadata.write_text(
        metadata_payload,
        encoding="utf-8",
    )

    destination_changed = bool(
        destination.is_file() and sha256(destination) != candidate_hash
    )
    if backup_path and not backup_path.exists():
        shutil.copy2(destination, backup_path)
    if destination_changed:
        shutil.copy2(destination, rollback_destination)

    try:
        os.replace(temporary_destination, destination)
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        if destination_changed and rollback_destination.is_file():
            os.replace(rollback_destination, destination)
        raise
    finally:
        temporary_destination.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        rollback_destination.unlink(missing_ok=True)

    print(f"Model đang dùng: {destination}")
    print(f"SHA-256: {candidate_hash}")
    print(f"Confidence mặc định: {default_confidence:.2f}")
    if backup_path:
        print(f"Model cũ đã sao lưu: {backup_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
