from datasets import load_dataset
import os

# 1. Load the dataset
dataset = load_dataset("blanchon/OSCD_RGB")

# 2. Define the root and subdirectories
base_dir = "dataset/changes"
subdirs = ["image1", "image2", "mask"]

# Create the folders on your Arch machine
for folder in subdirs:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

print("🚀 Starting download and extraction...")

# 3. Loop and save to specific folders
for i, example in enumerate(dataset["train"]):
    # Use a consistent naming scheme for easy pairing later
    filename = f"pair_{i:03d}.png"

    # Save Image 1 (Time A)
    example["image1"].save(os.path.join(base_dir, "image1", filename))

    # Save Image 2 (Time B)
    example["image2"].save(os.path.join(base_dir, "image2", filename))

    # Save the Change Mask
    example["mask"].save(os.path.join(base_dir, "mask", filename))

print(f"✅ Success! Data saved to:")
print(f"   - {base_dir}/image1/")
print(f"   - {base_dir}/image2/")
print(f"   - {base_dir}/mask/")
