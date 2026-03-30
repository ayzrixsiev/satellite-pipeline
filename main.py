from pipeline.ingest import DataIngestor
from pipeline.transform import DataTransformer
from pipeline.dataset import GeoSynthDataset

# Get the data
dataset_ingestor = DataIngestor("data/tiff/train", "data/tiff/train_labels")
dataset_ingestor.make_pairs()
paired_data = dataset_ingestor.get_data()

# Set up settings
transformer = DataTransformer(resized_images=(512, 512))

# Run the dataset
my_dataset = GeoSynthDataset(data_pairs=paired_data, transformer=transformer)

if len(my_dataset) > 0:
    print(f"Success! Found {len(my_dataset)} pairs.")

    sample_img, sample_mask = my_dataset[0]

    print(f"Image Tensor Shape: {sample_img.shape}")  # Should be (3, 512, 512)
    print(f"Mask Tensor Shape: {sample_mask.shape}")  # Should be (1, 512, 512)
    print(f"Image Max Value: {sample_img.max()}")  # Should be 1.0
else:
    print("No pairs found. Check your paths!")
