"""
build_report.py

This script reads the current project state from disk and writes one browser
data bundle to `frontend/report-data.js`.

The frontend expects a public dashboard-oriented shape:
- compact KPI values
- dataset registry rows
- chart series for histories
- latest metric snapshot
- featured qualitative panels
- system registry entries
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JS = REPO_ROOT / "frontend" / "report-data.js"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pipeline.ingest import ChangeDetectionIngestor, SegmentationIngestor


SUPPORTED_ARTIFACT_SUFFIXES = {".png", ".jpg", ".jpeg", ".pt", ".pth", ".json", ".csv"}


def _read_json(path: Path):
    """Return parsed JSON, or an empty structure when the file is missing."""

    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_metric(history: list[dict], key: str, default: float = 0.0) -> float:
    """Pick the latest value of one metric from an epoch history list."""

    if not history:
        return default
    value = history[-1].get(key, default)
    return float(value)


def _series_from_epoch_history(history: list[dict], key: str) -> list[float]:
    """Convert a list of epoch dictionaries into a single chart-ready series."""

    return [float(epoch.get(key, 0.0)) for epoch in history]


def _normalize_history(raw_history) -> dict[str, list[float]]:
    """Support both saved formats: list-of-epochs or dict-of-arrays."""

    if isinstance(raw_history, list):
        return {
            "train_loss": _series_from_epoch_history(raw_history, "train_loss"),
            "val_loss": _series_from_epoch_history(raw_history, "val_loss"),
            "train_dice": _series_from_epoch_history(raw_history, "train_dice"),
            "val_dice": _series_from_epoch_history(raw_history, "val_dice"),
            "epoch_rows": raw_history,
        }

    if isinstance(raw_history, dict):
        return {
            "train_loss": list(raw_history.get("train_loss", raw_history.get("loss", []))),
            "val_loss": list(raw_history.get("val_loss", [])),
            "train_dice": list(raw_history.get("train_dice", raw_history.get("dice", []))),
            "val_dice": list(raw_history.get("val_dice", [])),
            "epoch_rows": [],
        }

    return {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "epoch_rows": [],
    }


def _dataset_rows() -> list[dict]:
    """Build the exact dataset counts the backend actually uses."""

    roads_ingestor = SegmentationIngestor(
        train_images_dir=REPO_ROOT / "dataset" / "roads" / "train",
        train_masks_dir=REPO_ROOT / "dataset" / "roads" / "train_labels",
        val_images_dir=REPO_ROOT / "dataset" / "roads" / "val",
        val_masks_dir=REPO_ROOT / "dataset" / "roads" / "val_labels",
        split_seed=42,
    )
    roads_train, roads_val = roads_ingestor.build_splits()

    water_ingestor = SegmentationIngestor(
        train_images_dir=REPO_ROOT / "dataset" / "water_land" / "train",
        train_masks_dir=REPO_ROOT / "dataset" / "water_land" / "train_labels",
        split_seed=42,
    )
    water_train, water_val = water_ingestor.build_splits(val_ratio=0.2)

    change_ingestor = ChangeDetectionIngestor(
        train_image1_dir=REPO_ROOT / "dataset" / "changes" / "image1",
        train_image2_dir=REPO_ROOT / "dataset" / "changes" / "image2",
        train_masks_dir=REPO_ROOT / "dataset" / "changes" / "mask",
        split_seed=42,
    )
    change_train, change_val = change_ingestor.build_splits(val_ratio=0.2)

    return [
        {"name": "roads", "train": len(roads_train), "val": len(roads_val), "status": "READY"},
        {"name": "water_land", "train": len(water_train), "val": len(water_val), "status": "READY"},
        {"name": "changes", "train": len(change_train), "val": len(change_val), "status": "READY"},
    ]


def _list_prediction_images(task_name: str, prefix: str) -> list[str]:
    """Return browser-friendly relative paths for saved preview panels."""

    folder = OUTPUTS_ROOT / "predictions" / task_name
    if not folder.exists():
        return []

    files = sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() == ".png"
    )
    return [f"../outputs/predictions/{task_name}/{file_name}" for file_name in files]


def _count_artifacts(folder: Path) -> int:
    """Count saved reports, checkpoints, and images in outputs."""

    if not folder.exists():
        return 0

    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_ARTIFACT_SUFFIXES:
            total += 1
    return total


def _checkpoint_names() -> list[str]:
    """List the current checkpoint files for registry display."""

    checkpoint_dir = OUTPUTS_ROOT / "checkpoints"
    if not checkpoint_dir.exists():
        return []

    return sorted(path.name for path in checkpoint_dir.iterdir() if path.suffix.lower() in {".pt", ".pth"})


def _checkpoint_by_prefix(prefix: str) -> str:
    """Find the first checkpoint file that starts with a known task prefix."""

    for name in _checkpoint_names():
        if name.startswith(prefix):
            return name
    return "not found"


def build_report() -> dict:
    """Build the final payload consumed by the static frontend."""

    dataset_rows = _dataset_rows()
    total_samples = sum(dataset["train"] + dataset["val"] for dataset in dataset_rows)

    raw_roads_history = _read_json(OUTPUTS_ROOT / "reports" / "roads_history.json")
    raw_changes_history = _read_json(OUTPUTS_ROOT / "reports" / "changes_change_history.json")

    roads_history = _normalize_history(raw_roads_history)
    changes_history = _normalize_history(raw_changes_history)

    roads_metrics = {
        "epochs": len(roads_history["val_loss"]) or len(roads_history["train_loss"]),
        "val_loss": _latest_metric(roads_history["epoch_rows"], "val_loss", roads_history["val_loss"][-1] if roads_history["val_loss"] else 0.0),
        "val_iou": _latest_metric(roads_history["epoch_rows"], "val_iou", 0.0),
        "val_dice": _latest_metric(roads_history["epoch_rows"], "val_dice", roads_history["val_dice"][-1] if roads_history["val_dice"] else 0.0),
        "val_precision": _latest_metric(roads_history["epoch_rows"], "val_precision", 0.0),
        "val_recall": _latest_metric(roads_history["epoch_rows"], "val_recall", 0.0),
    }

    changes_metrics = {
        "epochs": len(changes_history["val_loss"]) or len(changes_history["train_loss"]),
        "val_loss": _latest_metric(changes_history["epoch_rows"], "val_loss", changes_history["val_loss"][-1] if changes_history["val_loss"] else 0.0),
        "val_iou": _latest_metric(changes_history["epoch_rows"], "val_iou", 0.0),
        "val_dice": _latest_metric(changes_history["epoch_rows"], "val_dice", changes_history["val_dice"][-1] if changes_history["val_dice"] else 0.0),
        "val_precision": _latest_metric(changes_history["epoch_rows"], "val_precision", 0.0),
        "val_recall": _latest_metric(changes_history["epoch_rows"], "val_recall", 0.0),
    }

    road_previews = _list_prediction_images("roads", "roads_preview")
    change_previews = _list_prediction_images("changes", "changes_preview")

    checkpoint_names = _checkpoint_names()
    checkpoint_count = len(checkpoint_names)
    artifact_count = _count_artifacts(OUTPUTS_ROOT)
    preview_count = len(road_previews) + len(change_previews)

    generated = datetime.now().isoformat(timespec="seconds")

    return {
        "generated": generated,
        "kpi": {
            "task_families": 2,
            "datasets": len(dataset_rows),
            "tracked_samples": total_samples,
            "checkpoints": checkpoint_count,
            "preview_panels": preview_count,
            "artifacts": artifact_count,
        },
        "datasets": dataset_rows,
        "histories": {
            "roads": {
                "train_loss": roads_history["train_loss"],
                "val_loss": roads_history["val_loss"],
                "train_dice": roads_history["train_dice"],
                "val_dice": roads_history["val_dice"],
            },
            "changes": {
                "train_loss": changes_history["train_loss"],
                "val_loss": changes_history["val_loss"],
                "train_dice": changes_history["train_dice"],
                "val_dice": changes_history["val_dice"],
            },
        },
        "metrics": {
            "roads": roads_metrics,
            "changes": changes_metrics,
        },
        "predictions": {
            "roads": road_previews,
            "changes": change_previews,
        },
        "registry": [
            {"key": "Platform", "val": "GeoSynth", "sub": "remote sensing dashboard"},
            {"key": "Task Modes", "val": "Segmentation + Change", "sub": "shared backend"},
            {"key": "Road Checkpoint", "val": _checkpoint_by_prefix("roads"), "sub": "current best file"},
            {"key": "Change Checkpoint", "val": _checkpoint_by_prefix("changes"), "sub": "current best file"},
            {"key": "Prediction Panels", "val": str(preview_count), "sub": "saved qualitative outputs"},
            {"key": "Histories", "val": "2", "sub": "roads and changes"},
            {"key": "Generated", "val": generated.replace("T", " "), "sub": "build_report.py"},
            {"key": "Serving", "val": "frontend/", "sub": "static local page"},
        ],
    }


def main() -> None:
    """Write the JS data bundle used by the browser dashboard."""

    report = build_report()
    js_content = (
        "// Auto-generated by frontend/build_report.py. Rebuild instead of editing.\n"
        "window.GEOSYNTH_REPORT = "
        + json.dumps(report, indent=2)
        + ";\n"
    )
    OUTPUT_JS.write_text(js_content, encoding="utf-8")

    print(f"[ OK ] report-data.js written to: {OUTPUT_JS}")
    print(f"       Tracked samples: {report['kpi']['tracked_samples']}")
    print(f"       Checkpoints: {report['kpi']['checkpoints']}")
    print(f"       Preview panels: {report['kpi']['preview_panels']}")


if __name__ == "__main__":
    main()
