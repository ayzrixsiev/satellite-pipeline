"""Build the browser report data from the current project state.

The frontend itself is static HTML/CSS/JS, which is great for screenshots and
easy sharing. This script prepares one JavaScript file that contains:
- dataset counts
- experiment metrics from JSON histories
- image paths for prediction panels
- project summary text for the report page

You can rerun this script any time after training to refresh the page data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys


# The script lives in `frontend/`, so two parents up lands at the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.ingest import ChangeDetectionIngestor, SegmentationIngestor


def _load_history(history_path: Path) -> list[dict]:
    """Load one metrics history file if it exists, otherwise return an empty list."""

    if not history_path.exists():
        return []

    return json.loads(history_path.read_text(encoding="utf-8"))


def _relative_to_frontend(path: Path) -> str:
    """Convert a project path into a path that works from `frontend/index.html`."""

    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _count_existing(paths: list[Path]) -> int:
    """Count how many files from a list actually exist on disk."""

    return sum(1 for path in paths if path.exists())


def _dataset_inventory() -> dict:
    """Collect dataset counts from the real folders on disk."""

    roads_ingestor = SegmentationIngestor(
        train_images_dir=PROJECT_ROOT / "dataset" / "roads" / "train",
        train_masks_dir=PROJECT_ROOT / "dataset" / "roads" / "train_labels",
        val_images_dir=PROJECT_ROOT / "dataset" / "roads" / "val",
        val_masks_dir=PROJECT_ROOT / "dataset" / "roads" / "val_labels",
        split_seed=42,
    )
    roads_train, roads_val = roads_ingestor.build_splits()

    water_ingestor = SegmentationIngestor(
        train_images_dir=PROJECT_ROOT / "dataset" / "water_land" / "train",
        train_masks_dir=PROJECT_ROOT / "dataset" / "water_land" / "train_labels",
        split_seed=42,
    )
    water_train, water_val = water_ingestor.build_splits(val_ratio=0.2)

    change_ingestor = ChangeDetectionIngestor(
        train_image1_dir=PROJECT_ROOT / "dataset" / "changes" / "image1",
        train_image2_dir=PROJECT_ROOT / "dataset" / "changes" / "image2",
        train_masks_dir=PROJECT_ROOT / "dataset" / "changes" / "mask",
        split_seed=42,
    )
    change_train, change_val = change_ingestor.build_splits(val_ratio=0.2)

    return {
        "roads": {
            "title": "Road Detection",
            "status": "Baseline trained",
            "train_samples": len(roads_train),
            "val_samples": len(roads_val),
            "notes": "GeoTIFF roads dataset with explicit validation split.",
        },
        "water_land": {
            "title": "Land vs Water",
            "status": "Dataset integrated",
            "train_samples": len(water_train),
            "val_samples": len(water_val),
            "notes": "Masks are JPEG files, so the backend binarizes them defensively.",
        },
        "changes": {
            "title": "Change Detection",
            "status": "Baseline trained",
            "train_samples": len(change_train),
            "val_samples": len(change_val),
            "notes": "Paired `image1/image2/mask` dataset using a 6-channel input baseline.",
        },
    }


def _experiment_cards() -> list[dict]:
    """Create the high-level report cards shown at the top of the webpage."""

    roads_history = _load_history(PROJECT_ROOT / "outputs" / "reports" / "roads_history.json")
    changes_history = _load_history(PROJECT_ROOT / "outputs" / "reports" / "changes_change_history.json")
    inventory = _dataset_inventory()
    total_samples = sum(item["train_samples"] + item["val_samples"] for item in inventory.values())

    artifact_paths = [
        PROJECT_ROOT / "outputs" / "reports" / "roads_history.json",
        PROJECT_ROOT / "outputs" / "reports" / "changes_change_history.json",
        PROJECT_ROOT / "outputs" / "checkpoints" / "roads_best.pth",
        PROJECT_ROOT / "outputs" / "checkpoints" / "changes_change_best.pth",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_1.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_2.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_prediction_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_preview_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_preview_1.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_prediction_0.png",
    ]

    return [
        {
            "label": "Supported task families",
            "value": "2",
            "detail": "Segmentation and paired-image change detection",
        },
        {
            "label": "Integrated datasets",
            "value": "3",
            "detail": "Roads, land/water, and change detection",
        },
        {
            "label": "Labeled samples tracked",
            "value": f"{total_samples}",
            "detail": "Train and validation samples across all wired datasets",
        },
        {
            "label": "Saved report artifacts",
            "value": f"{_count_existing(artifact_paths)}",
            "detail": "Histories, checkpoints, and qualitative prediction panels",
        },
        {
            "label": "Baseline runs completed",
            "value": str(sum(1 for history in [roads_history, changes_history] if history)),
            "detail": "Road detection and change detection both have working training histories",
        },
        {
            "label": "Prediction panels linked",
            "value": str(len(_gallery_items())),
            "detail": "Visual outputs are already connected to the browser report gallery",
        },
    ]


def _gallery_items() -> list[dict]:
    """Collect qualitative prediction panels for the report gallery."""

    gallery_paths = [
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_1.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_preview_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_prediction_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_prediction_0.png",
    ]

    items: list[dict] = []
    for path in gallery_paths:
        if not path.exists():
            continue

        task_name = "Change Detection" if "changes" in path.parts else "Road Detection"
        items.append(
            {
                "title": path.stem.replace("_", " ").title(),
                "task": task_name,
                "image": f"../{_relative_to_frontend(path)}",
            }
        )

    return items


def _artifact_inventory() -> list[dict]:
    """Summarize the concrete files produced by the backend so far."""

    reports = [
        PROJECT_ROOT / "outputs" / "reports" / "roads_history.json",
        PROJECT_ROOT / "outputs" / "reports" / "changes_change_history.json",
    ]
    checkpoints = [
        PROJECT_ROOT / "outputs" / "checkpoints" / "roads_best.pth",
        PROJECT_ROOT / "outputs" / "checkpoints" / "changes_change_best.pth",
    ]
    previews = [
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_1.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_preview_2.png",
        PROJECT_ROOT / "outputs" / "predictions" / "roads" / "roads_prediction_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_preview_0.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_preview_1.png",
        PROJECT_ROOT / "outputs" / "predictions" / "changes" / "changes_prediction_0.png",
    ]

    return [
        {
            "title": "Metric histories",
            "value": str(_count_existing(reports)),
            "detail": "JSON files saved after training runs and used directly by the frontend charts.",
        },
        {
            "title": "Model checkpoints",
            "value": str(_count_existing(checkpoints)),
            "detail": "Reusable trained weights for road segmentation and change detection baselines.",
        },
        {
            "title": "Qualitative panels",
            "value": str(_count_existing(previews)),
            "detail": "Saved prediction images for visual comparison in the gallery section.",
        },
    ]


def build_report_payload() -> dict:
    """Create the full payload consumed by the frontend JavaScript."""

    roads_history = _load_history(PROJECT_ROOT / "outputs" / "reports" / "roads_history.json")
    changes_history = _load_history(PROJECT_ROOT / "outputs" / "reports" / "changes_change_history.json")

    return {
        "project": {
            "title": "GeoSynth Mission Report",
            "subtitle": "Remote-sensing computer vision system for segmentation and change detection",
            "summary": (
                "This report page summarizes the current backend for GeoSynth: "
                "a reusable pipeline that ingests datasets, preprocesses imagery, "
                "trains baseline models, and exports prediction panels ready for a future frontend."
            ),
            "mentor_pitch": (
                "The project is already beyond a notebook prototype. It supports multiple "
                "dataset families, task-specific training entrypoints, reusable preprocessing, "
                "model checkpointing, metrics export, and polished qualitative outputs. "
                "The current scores are early baselines, but the engineering foundation is ready for iteration."
            ),
            "generated_on": datetime.now().astimezone().strftime("%B %d, %Y at %H:%M"),
        },
        "report_cards": _experiment_cards(),
        "inventory": _dataset_inventory(),
        "artifacts": _artifact_inventory(),
        "histories": {
            "roads": roads_history,
            "changes": changes_history,
        },
        "gallery": _gallery_items(),
        "pipeline_steps": [
            {
                "title": "1. Ingest",
                "body": "Pair segmentation images with masks or pair change-detection image1/image2/mask triplets by filename stem.",
            },
            {
                "title": "2. Transform",
                "body": "Load TIFF, PNG, and JPEG files, binarize masks safely, resize inputs, normalize tensors, and apply synchronized augmentations.",
            },
            {
                "title": "3. Train",
                "body": "Build task-specific dataloaders, run epoch loops, compute losses and metrics, and save the best checkpoint.",
            },
            {
                "title": "4. Evaluate",
                "body": "Track validation IoU and Dice over time and export machine-readable history files for later reporting.",
            },
            {
                "title": "5. Showcase",
                "body": "Save visual comparison panels and surface them in a browser-ready report page for mentor updates and demos.",
            },
        ],
        "achievements": [
            "Built a generic segmentation ingestor for both road and land/water datasets.",
            "Built a dedicated change-detection ingestor for paired temporal imagery.",
            "Implemented one shared preprocessing pipeline that handles TIFF, PNG, and JPEG masks safely.",
            "Added reusable training utilities for device selection, dataloaders, metrics, and checkpoints.",
            "Trained working baseline models for road segmentation and change detection.",
            "Exported qualitative prediction panels to support rapid visual review.",
        ],
        "next_steps": [
            "Train the water-vs-land benchmark and add its metrics to the dashboard.",
            "Improve the road baseline with more epochs, tuned class weighting, and stronger augmentation.",
            "Replace OpenCV-only TIFF loading with a GeoTIFF-aware reader for cleaner remote-sensing support.",
            "Add a lightweight browser UI for selecting tasks, images, and prediction runs interactively.",
        ],
        "run_commands": [
            "./.venv/bin/python main.py segmentation train --dataset roads --epochs 5 --image-size 256",
            "./.venv/bin/python main.py segmentation train --dataset water_land --epochs 5 --image-size 256",
            "./.venv/bin/python main.py change_detection train --dataset changes --epochs 8 --image-size 256",
        ],
    }


def main() -> None:
    """Write the payload as a small JavaScript file used by the browser page."""

    payload = build_report_payload()
    output_path = PROJECT_ROOT / "frontend" / "report-data.js"

    # Using a JS assignment instead of JSON fetch keeps the page easy to open
    # from a simple local server without any API layer.
    output_path.write_text(
        "window.GEOSYNTH_REPORT = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Report data written to {output_path}")


if __name__ == "__main__":
    main()
