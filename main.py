import os
import torch

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

from pipeline.ingest import DataIngestor
from pipeline.transform import DataTransformer
from pipeline.dataset import GeoSynthDataset
from torch.utils.data import DataLoader
from pipeline.train import train_one_epoch, model, optimizer, criterion, device
from visualize import save_prediction

# Get the data
dataset_ingestor = DataIngestor("data/tiff/train", "data/tiff/train_labels")
dataset_ingestor.make_pairs()
paired_data = dataset_ingestor.get_data()


# Set up settings
transformer = DataTransformer(resized_images=(512, 512))


# Run the dataset and check whether everything worked
my_dataset = GeoSynthDataset(data_pairs=paired_data, transformer=transformer)
# if len(my_dataset) > 0:
#     print(f"Success! Found {len(my_dataset)} pairs.")

#     sample_img, sample_mask = my_dataset[0]

#     print(f"Image Tensor Shape: {sample_img.shape}")  # Should be (3, 512, 512)
#     print(f"Mask Tensor Shape: {sample_mask.shape}")  # Should be (1, 512, 512)
#     print(f"Image Max Value: {sample_img.max()}")  # Should be 1.0
# else:
#     print("No pairs found.")


train_loader = DataLoader(
    my_dataset,
    batch_size=4,  # 4 images at a time
    shuffle=True,
    num_workers=2,  # Uses my Arch CPU to prep data in parallel
)

# Run for 10 'Eras' (Epochs)
for epoch in range(10):
    avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

save_prediction(model, my_dataset, 0, device)
torch.save(model.state_dict(), "geosynth_roads.pth")
print("Model weights saved to geosynth_roads.pth")
