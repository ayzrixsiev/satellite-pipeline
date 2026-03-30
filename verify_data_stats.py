import cv2
import numpy as np
import os

IMAGE_PATH = "./data/tiff/train/25829245_15.tiff"
MASK_PATH = "./data/tiff/train_labels/10078660_15.tif"


def scout():
    img = cv2.imread(IMAGE_PATH)
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print("Error: Could not find the files.")
        return

    print(f"--- IMAGE STATS ---")
    print(f"Shape: {img.shape}")
    print(f"Data Type: {img.dtype}")
    print(f"Min/Max Pixel Values: {np.min(img)} / {np.max(img)}")

    print(f"\n--- MASK STATS ---")
    print(f"Shape: {mask.shape}")
    print(f"Unique Values in Mask: {np.unique(mask)}")
    print(f"Data Type: {mask.dtype}")


if __name__ == "__main__":
    scout()
