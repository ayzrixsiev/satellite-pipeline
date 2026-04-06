# Project GeoSythn
System that ingests a dataset with satelite and drone taken images to detect crop type in Uzbekistan. It runs a whole pipeline to clean and creates one format, comfortable to feed and train machine learning model which will, then, detect crop type and make statistics report automatically.

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

