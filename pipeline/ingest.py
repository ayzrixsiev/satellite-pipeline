"""Ingest sorted into pairs dataset into the system"""

import os
import logging

# Logger to use instead of print
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DataIngestor:

    # Get the path to the dataset and make list of images
    def __init__(self, images_dir, masks_dir):

        # Store the DIRECTORY paths themselves
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # Store the list of FILENAMES
        self.image_filenames = os.listdir(images_dir)
        self.mask_filenames = os.listdir(masks_dir)

        self.pair_sorted_data = []

    # Make pairs of an image and it's corresponding mask
    def sort_data(self):
        self.pair_sorted_data = []
        dict_of_masks = {os.path.splitext(f)[0]: f for f in self.mask_filenames}
        for img in self.image_filenames:
            image_stem = os.path.splitext(img)[0]

            if image_stem in dict_of_masks:
                matching_mask = dict_of_masks[image_stem]

                full_image_path = os.path.join(self.images_dir, img)
                full_mask_path = os.path.join(self.masks_dir, matching_mask)

                if (
                    os.path.getsize(full_image_path) > 0
                    and os.path.getsize(full_mask_path) > 0
                ):
                    self.pair_sorted_data.append((full_image_path, full_mask_path))

                else:
                    logger.warning(f"Skipping corrupted file: {img}")

    def get_sorted_data(self):
        return self.pair_sorted_data


func = DataIngestor("../data/tiff/train", "../data/tiff/train_labels")
func.sort_data()
result = func.get_sorted_data()
print(len(result))
