# Project
System that ingests a dataset with satelite and drone taken images to detect a roards. It runs a whole pipeline to clean and creates one format, comfortable to feed and train machine learning model which will, then, detect roads automatically and showcase the results

## Libraries and their roles in my system
1. torch & torchvision - the engine: These will use your GTX 1650 to do the heavy lifting
2. numpy - the translator: Converts raw images into numerical matrices
3. opencv-python - the vision: Handles reading .tiff files and drawing the road masks
4. segmentation-models-pytorch - the architect: Provides the U-Net architecture so we don't reinvent the wheel
5. pydantic - the validator: Ensures that if you drop in a new dataset, it matches the required format/schema (size, channels, etc.)
7. albumentations - the augmentor: Automatically flips/rotates images to make the model smarter

## High level design and overview
1. Ingest - read the data using opencv
2. Clean/Standartization - clean the data with numpy/albumentation and create one standart form with pydantic
3. Training - train the model on the cleaned and comfortable format with Pytorch and segmentation-models-pytorch
4. Show - present a visual outcomes

## Additional packages
1. uv - extremely fast python manager written in rust
2. uvicorn+starlette - server for fastapi

## Work to do and outcomes
- We need to see how our data looks like in the first place. What size images are? Are they grayscale (0-255) or they binary (0, 1). Our images are 1500 by 1500 which is high resolution for my machine, our mask data has only black and white colors, which is very comfortable

- We need to cut those big images into 512x512 pieces to work with them, meaning once we ingest we need to start cutting them and store, this will be normalization step.



## Knowledge
Grayscale: Reducing 3 color channels (Red, Green, Blue) into 1 channel (Brightness).
Normalization: Changing the range of the numbers.