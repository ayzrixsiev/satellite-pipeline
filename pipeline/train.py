import segmentation_models_pytorch as smp
import torch

# Declare a model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,  # Matching my [3, 512, 512]
    classes=1,  # Matching my [1, 512, 512]
)

# Move it to my GTX 1650
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the "Judge". How wrong am i?
criterion = torch.nn.BCEWithLogitsLoss()

# Set up the "Teacher". How to fix my mistakes? We use Adam because it is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model is loaded on: {device}")


def train_one_epoch(model, loader, optimizer, criterian, device):
    model.train()
    epoch_loss = 0

    for batch in loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Clear the memory after previous 4 batches/images
        outputs = model(images)  # Giving model images and training it

        loss = criterian(
            outputs, labels
        )  # Compare the labels to the gueses that U Net made

        loss.backward()  # Backpropagation - which means we are checking why we are wrong on these cases
        optimizer.step()  # Now when we know why we are wrong, we tweak the weights so that next time we are closer to the thruth

        # All the mistake scores
        epoch_loss += loss.item()

        # Find how wrong was the model in percentages
        return epoch_loss / len(loader)
