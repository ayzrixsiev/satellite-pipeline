- LOSS is how confident the model is in it's wrong predictions, the higher confidence the higher loss number
- VAL_LOSS is for images that model did not seen, if val loss is going up and loss down, that means model is overfit and 
- ACURACY is how well the model performed based on the amount of pixels it got wrong (potential problem is that model can be good at predicting land, which is 70% of the image and poorly on roads, which takes smaller space)
- JACCARD COEFFICIENT (intersection over union) is how close predicted shape is to the actual shape.
- VAL_JACCARD is the utlimate test in our semantic segmentation, IoU over unseen data
- LEARNING RATE is the metric to see how big the changes are in weights during iterations



I am working on Dubai dataset that obtained by MBRSC satellites, annotated with pixel wise semantic segmentation in 6 classes. I have 72 images, in 8 large tiles.

Classes to detect:
1. Building: #3C1098
2. Land (unpaved area): #8429F6
3. Road: #6EC1E4
4. Vegetation: #FEDD3A
5. Water: #E2A929
6. Unlabeled: #9B9B9B

**First**. I did is cropping and then cutting images into 256x256 tiles, i scalled them into 0-1 for future math. Here is the result i got:

images: 1305
masks: 1305
images dataset shape: (1305, 256, 256, 3)
masks dataset shape: (1305, 256, 256, 3)

**Second**. I did was converting labels from hex to rgb for future model analysis. Here are the rgb results i got for each class:
1. Building: [ 60  16 152]
2. Land:     [132  41 246]
3. Road:     [110 193 228]
4. Veg:      [254 221  58]
5. Water:    [226 169  41]
6. Unlabeled:[155 155 155]

**Third**. I did was creating "label images", meaning i had an empty canvas where i placed numbers, each number represented a certain class as integer from 0 to 5. I did it because U-NET model will create a probability vector for each pixel. For example: [0.05, 0.85, 0.02, 0.01, 0.02, 0.05], in this case class 1 has the higest probability - 85%, during training our evaluation metrics will go through this predictions and ground thruth, and compare the results, and then update the weights.
I also add additional channel dimension, because keras and tensorflow needs to work with: (Batch Size, Height, Width, Channels) structure. By adding that extra dimension at axis=3, i turn the shape into (Num_Images, 256, 256, 1).

**Fourth**. Now i have the following lable structure: (256, 256, 1). But U-NET model outputs an image as (256, 256, 6), matching our 6 class structure (where channel 0 is water, 3 is buildings). For us to calculate the loss (error), we need to subtract prediction from ground thruth, but you can not mathematically subtract 6 channel from 1 chanell and because of that i converted it to one-hot-encoding. Additionally, model can treat this array of integers [0, 1, 3, 4, 5] incorrectly. For example, it may think that class 3 buildings are greater than class 0 water, it might also think that if it mistakes a Building (3) for a Road (2), it’s a "better" mistake than mistaking it for Water (0) because the numbers are closer. One-hot-encoding converts these classes into independent dimensions: 
Class 1 becomes: [0, 1, 0, 0, 0, 0]     
Class 5 becomes: [0, 0, 0, 0, 0, 1]     
After conversion i split the data into classic 80 train/20 test.

**Fifth**. I created Jaccard Coefficient AKA Intersection over Union (IoU). Our U-NET model predicts shapes, boundaries in the image, we take those predictions (as image), and unite together with ground truth and then check where they intersect. While pixel accuracy only checks how many pixels you got right, the Jaccard Coefficient measures how well the predicted shapes overlap with the actual shapes.

**Sixth**. I created a custom architecture of encoder (down sampling)/decoder (up sampling) layers for U-NET model. If we look at satellite or aerial images, each pixel will represent a whole lake or land. We can clearly see the objects and where they are located, but we do not have details on exactly what it is. This is why we have encoder that compresses the data, which you can think of as "squint", you try to look at the details more clearly and more "zoomed in like". We do this to find details and clearly understand what we have on the image. We apply this operation several times, and more we compress the image, the more kernels we slide on the image. Now once its done, model knows that this is lake, this is land and etc. But images looks like a giant blocky square, because we sacrifised spatial accuracy (the exact where) to achieve a deep understanding of the "what" (global context). Decoder helps us fix this issue, by up sampling the details back, from 16x16 to 32x32 and etc, and it does that untill we reach original size.

**Seventh**. I defined evaluation metrics, first two - accuracy and jaccard coefficient is for me human. 
1. Accuracy is a simple metric that just checks each predicted pixel against ground truth, it is not the best metric to check the performance of a model though, because if your dataset is dominant lets say with land, model learns to predict it easily, while other classes it may predict poorly, but because there is more land, more pixels are predicted right, hence accuracy is high. 
2. Jaccard coefficient is used to see IoU.

After that i implemented two different loss functions that is for the model.        
1. Dice loss (shape punisher), it measures overlap between predicted shape and ground truth and puneshes the model is shape is not predicted properly, even if borders are slightly off.
2. Categorical Focal loss (hard class spotlight), it applies a mathematical modifier that dynamically scales down the loss of easy, highly-confident pixels. If the model says, "I am 99% sure this pixel is land," Focal Loss reduces that pixel's importance to almost zero. If the model says, "I have no idea what this road pixel is," Focal Loss magnifies that error. This forces the U-Net to stop spending computational energy on what it already knows, and forces it to focus entirely on learning hard features, like road lines and building edges. And then i combine them both into one: total loss.        
Finally i set up Adam for changing the weights, four other metrics.

**Final**. We have Adam that adjusts weights during training. It has something called learning rate, which defines how aggressive the model changes weights values. I set up learning rate scheduler, which watches validation loss (model trying to predict after each epoch) and if it is not improving for 5 epochs, it cuts down learning rate by two, it helps model start making softer/smaller adjustments which results in better mapping small tricky details. Another function i set up is Early stopping. Which watches that validation loss, and if after 12 epochs (2 lr cuts), it is not improving, it means that the model will just start overfitting and remembering pixels, which will lead to performance to get worse, so we stop the training and get back to the best result we had so far. Finally i started 100 epoches training.