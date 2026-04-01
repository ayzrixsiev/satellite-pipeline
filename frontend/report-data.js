window.GEOSYNTH_REPORT = {
  "project": {
    "title": "GeoSynth Mission Report",
    "subtitle": "Remote-sensing computer vision system for segmentation and change detection",
    "summary": "This report page summarizes the current backend for GeoSynth: a reusable pipeline that ingests datasets, preprocesses imagery, trains baseline models, and exports prediction panels ready for a future frontend.",
    "mentor_pitch": "The project is already beyond a notebook prototype. It supports multiple dataset families, task-specific training entrypoints, reusable preprocessing, model checkpointing, metrics export, and polished qualitative outputs. The current scores are early baselines, but the engineering foundation is ready for iteration.",
    "generated_on": "April 01, 2026 at 20:44"
  },
  "report_cards": [
    {
      "label": "Supported task families",
      "value": "2",
      "detail": "Segmentation and paired-image change detection"
    },
    {
      "label": "Integrated datasets",
      "value": "3",
      "detail": "Roads, land/water, and change detection"
    },
    {
      "label": "Labeled samples tracked",
      "value": "2941",
      "detail": "Train and validation samples across all wired datasets"
    },
    {
      "label": "Saved report artifacts",
      "value": "11",
      "detail": "Histories, checkpoints, and qualitative prediction panels"
    },
    {
      "label": "Baseline runs completed",
      "value": "2",
      "detail": "Road detection and change detection both have working training histories"
    },
    {
      "label": "Prediction panels linked",
      "value": "5",
      "detail": "Visual outputs are already connected to the browser report gallery"
    }
  ],
  "inventory": {
    "roads": {
      "title": "Road Detection",
      "status": "Baseline trained",
      "train_samples": 72,
      "val_samples": 14,
      "notes": "GeoTIFF roads dataset with explicit validation split."
    },
    "water_land": {
      "title": "Land vs Water",
      "status": "Dataset integrated",
      "train_samples": 2273,
      "val_samples": 568,
      "notes": "Masks are JPEG files, so the backend binarizes them defensively."
    },
    "changes": {
      "title": "Change Detection",
      "status": "Baseline trained",
      "train_samples": 12,
      "val_samples": 2,
      "notes": "Paired `image1/image2/mask` dataset using a 6-channel input baseline."
    }
  },
  "artifacts": [
    {
      "title": "Metric histories",
      "value": "2",
      "detail": "JSON files saved after training runs and used directly by the frontend charts."
    },
    {
      "title": "Model checkpoints",
      "value": "2",
      "detail": "Reusable trained weights for road segmentation and change detection baselines."
    },
    {
      "title": "Qualitative panels",
      "value": "7",
      "detail": "Saved prediction images for visual comparison in the gallery section."
    }
  ],
  "histories": {
    "roads": [
      {
        "epoch": 1,
        "train_loss": 0.5172121334407065,
        "train_precision": 0.024893975280693027,
        "train_recall": 0.13222741779851102,
        "train_iou": 0.021398114227372443,
        "train_dice": 0.0418996548540896,
        "val_loss": 0.6867423057556152,
        "val_precision": 0.10227272726110538,
        "val_recall": 0.0005743092336158673,
        "val_iou": 0.0005714285714282086,
        "val_dice": 0.001142204454596648
      }
    ],
    "changes": [
      {
        "epoch": 1,
        "train_loss": 0.5045738418896993,
        "train_precision": 0.029004189494029532,
        "train_recall": 0.2499999999993687,
        "train_iou": 0.026682478505774015,
        "train_dice": 0.05197805371064186,
        "val_loss": 0.9090759754180908,
        "val_precision": 0.018175426261199445,
        "val_recall": 0.9315315315147472,
        "val_iou": 0.01815117789558047,
        "val_dice": 0.03565517241378081
      }
    ]
  },
  "gallery": [
    {
      "title": "Roads Preview 0",
      "task": "Road Detection",
      "image": "../outputs/predictions/roads/roads_preview_0.png"
    },
    {
      "title": "Roads Preview 1",
      "task": "Road Detection",
      "image": "../outputs/predictions/roads/roads_preview_1.png"
    },
    {
      "title": "Changes Preview 0",
      "task": "Change Detection",
      "image": "../outputs/predictions/changes/changes_preview_0.png"
    },
    {
      "title": "Roads Prediction 0",
      "task": "Road Detection",
      "image": "../outputs/predictions/roads/roads_prediction_0.png"
    },
    {
      "title": "Changes Prediction 0",
      "task": "Change Detection",
      "image": "../outputs/predictions/changes/changes_prediction_0.png"
    }
  ],
  "pipeline_steps": [
    {
      "title": "1. Ingest",
      "body": "Pair segmentation images with masks or pair change-detection image1/image2/mask triplets by filename stem."
    },
    {
      "title": "2. Transform",
      "body": "Load TIFF, PNG, and JPEG files, binarize masks safely, resize inputs, normalize tensors, and apply synchronized augmentations."
    },
    {
      "title": "3. Train",
      "body": "Build task-specific dataloaders, run epoch loops, compute losses and metrics, and save the best checkpoint."
    },
    {
      "title": "4. Evaluate",
      "body": "Track validation IoU and Dice over time and export machine-readable history files for later reporting."
    },
    {
      "title": "5. Showcase",
      "body": "Save visual comparison panels and surface them in a browser-ready report page for mentor updates and demos."
    }
  ],
  "achievements": [
    "Built a generic segmentation ingestor for both road and land/water datasets.",
    "Built a dedicated change-detection ingestor for paired temporal imagery.",
    "Implemented one shared preprocessing pipeline that handles TIFF, PNG, and JPEG masks safely.",
    "Added reusable training utilities for device selection, dataloaders, metrics, and checkpoints.",
    "Trained working baseline models for road segmentation and change detection.",
    "Exported qualitative prediction panels to support rapid visual review."
  ],
  "next_steps": [
    "Train the water-vs-land benchmark and add its metrics to the dashboard.",
    "Improve the road baseline with more epochs, tuned class weighting, and stronger augmentation.",
    "Replace OpenCV-only TIFF loading with a GeoTIFF-aware reader for cleaner remote-sensing support.",
    "Add a lightweight browser UI for selecting tasks, images, and prediction runs interactively."
  ],
  "run_commands": [
    "./.venv/bin/python main.py segmentation train --dataset roads --epochs 5 --image-size 256",
    "./.venv/bin/python main.py segmentation train --dataset water_land --epochs 5 --image-size 256",
    "./.venv/bin/python main.py change_detection train --dataset changes --epochs 8 --image-size 256"
  ]
};
