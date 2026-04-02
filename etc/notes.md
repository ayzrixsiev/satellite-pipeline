# Project GeoSythn
System that ingests a dataset with satelite and drone taken images to detect a roards, diff water and land and find change over time in the images. It runs a whole pipeline to clean and creates one format, comfortable to feed and train machine learning model which will, then, detect roads automatically and showcase the results

## Libraries and their roles in my system
1. torch & torchvision - the engine: these will use your GTX 1650 to do the heavy lifting
2. numpy - the translator: converts raw images into numerical matrices
3. opencv-python - the vision: handles reading files
4. segmentation-models-pytorch - the architect: provides the U-Net architecture (ML model)
5. pydantic - the validator: Ensures that if you drop in a new dataset, it matches the required format/schema (size, channels, etc.)
7. albumentations - the augmentor: automatically flips/rotates images to make the model smarter

## High level design and overview
1. Ingest - read the data using opencv
2. Clean/Standartization - clean the data with numpy/albumentation and create one standart form with pydantic
3. Training - train the model on the cleaned and comfortable format with Pytorch and segmentation-models-pytorch
4. Show - present a visual outcomes

## Additional packages
1. uv - extremely fast python manager written in rust
2. uvicorn+starlette - server for fastapi


## Work done, outcomes and what i have learned

- Ingesting datasets, we have two DataIngestors - for segmenation and for change detection datasets. Three datasets: for road detection, for water-land detection, for change detection. We validate pathes, then we pair images (image and it's mask) using "stem", we split the dataset into two parts (if it is not split) for training and validation (usually 20 val and 80 train).

- Transform datasets, we process images and masks by making them same size, opencv loads images with bgr we make it rgb, we make sure masks are grayscale and their range values bigger (basically masks are usually 1 white roi and 0 black, and we make this 1 be 255 to be able to see it visually), we also make sure masks and images are the same size. I have added an option of augmentation - this is a technique used to flip an images, so that ai does not just remembers the images, but actually can detect roads from any angle in the images. And then we orchestrate all the _files in two callable functions.