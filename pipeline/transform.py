"""Make data comfortable to work with and to feed to ML model"""

import cv2
import numpy as np


class DataTransformer:
    def __init__(self, resized_images=(512, 512)):
        self.resized_images = resized_images

    def process_images(self, img_path):
        img = cv2.imread(img_path)

        # Convert BGR to RGB (AI standard)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize (tailored to my machine)
        img = cv2.resize(img, self.resized_images)

        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0

        return img

    def process_labels(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image and make sure it is it is binary
        mask = cv2.resize(mask, self.resized_images, interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Normilize to 0-1
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return mask
