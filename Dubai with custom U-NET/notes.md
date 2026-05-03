I am training U-NET model on the Dubai dataset which has both images and masks. Initially i started to crop all the images and cut them into 256 by 256 patches to feed a model. then i scalled them into 0-1. We also have hex values of all objects on the images, which i am converting into one hot encoding format (which is essentially telling a model what each color on the mask means in terms of object: Water, Building and etc, besides that it prevents potential errors like making model think Class 1 (building) is smaller than Class 3 (water) and etc). I created labels dataset (based on the masks) which creates images that consists of numbers between 0-5, each of this number is the representation of an objects on the image (building, water and etc). Here is the workflow of training:

1. Feed the Image (X) into the U-Net.
2. U-Net guesses what the labels should be (this is called the prediction).
3. Compare the Prediction to the Labels (Y).
4. Calculate the "Loss" (the score of how wrong the AI was).
5. Optimizer tweaks the U-Net's neurons to reduce that Loss.