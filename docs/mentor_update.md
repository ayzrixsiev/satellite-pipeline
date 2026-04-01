# GeoSynth Progress Update

## Current project direction

GeoSynth is being developed as a reusable remote-sensing computer vision system rather than a collection of isolated notebooks. The current backend supports both binary segmentation and paired-image change detection.

## What has been completed

- Implemented a generic segmentation data ingestor that supports both road detection and land-vs-water datasets.
- Implemented a dedicated change-detection ingestor that pairs image1, image2, and mask files by filename stem.
- Added one shared transformation pipeline for reading images, binarizing masks, resizing data, normalizing tensors, and applying synchronized augmentations.
- Added reusable PyTorch dataset classes for segmentation and change-detection tasks.
- Added shared training utilities for dataloaders, metrics, checkpoints, and epoch loops.
- Trained first working baselines for road segmentation and change detection.
- Added qualitative prediction panels to visually compare model outputs with source imagery and labels.
- Built a browser-ready static report page that presents project status, metrics, and example outputs.

## Current integrated datasets

- Roads dataset with explicit train and validation folders
- Land-vs-water dataset with automatic train/validation split
- Change detection dataset with paired temporal images and binary masks

## What has been achieved technically

- The system can now support multiple task families through a shared backend structure.
- It can save checkpoints, metric histories, and qualitative prediction artifacts.
- It is already ready to support a future interactive frontend.

## Next steps

- Train the land-vs-water benchmark and add it to the report page
- Improve road detection performance with longer training and stronger augmentation
- Improve geospatial TIFF handling with a more dedicated TIFF reader
- Build an interactive browser UI on top of the current backend outputs

