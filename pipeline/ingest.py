"""Find a necessary data, check the data type (tiff) and then find image and it's label pair"""

import os


class DataIngestor:
    def __init__(self, raw_images, raw_labels):
        self.raw_images = raw_images
        self.raw_labels = raw_labels
        self.data_pair = []

    def make_pairs(self):
        image_files = os.listdir(self.raw_images)
        label_files = os.listdir(self.raw_labels)
        # Creating a dict of {"Stem": "Full file name"}, stem is a name before .tif
        label_map = {os.path.splitext(f)[0]: f for f in label_files}
        for image in image_files:
            image_stem = os.path.splitext(image)[
                0
            ]  # Get the name of an image without it's extension .tiff
            if image_stem in label_map:
                matching_label = label_map[image_stem]

                full_image_path = os.path.join(self.raw_images, image)
                full_label_path = os.path.join(self.raw_labels, matching_label)
                # Check if pathes are valid
                if (
                    os.path.getsize(full_image_path) > 0
                    and os.path.getsize(full_label_path) > 0
                ):
                    self.data_pair.append((full_image_path, full_label_path))
                else:
                    print(f"Skipping corrupted file: {image}")

                self.data_pair.append((full_image_path, full_label_path))

        return self.data_pair

    def get_data(self):
        return self.data_pair


data_ingestor = DataIngestor("../data/tiff/train", "../data/tiff/train_labels")
data_ingestor.make_pairs()
paired_data = data_ingestor.get_data()
print(paired_data)
