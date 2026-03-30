"""Make data comfortable to work with and to feed to the Machine Learning model"""

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

        # The Math: Convert to float and normalize
        img = img.astype(np.float32) / 255.0

        return img

    def process_labels(self, label_path):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Resize to the images size
        label = cv2.resize(label, self.resized_images)

        # Normilize to 0-1 range
        label = label.astype(np.float32) / 255.0

        return label
