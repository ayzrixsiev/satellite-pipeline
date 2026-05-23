# Version 1

I am working on a Dubai dataset obtained by MBRSC satellites, annotated with pixel-wise semantic segmentation in 6 classes. I have 72 images across 8 large tiles.

Classes to detect:

1. Building: #3C1098
2. Land (unpaved area): #8429F6
3. Road: #6EC1E4
4. Vegetation: #FEDD3A
5. Water: #E2A929
6. Unlabeled: #9B9B9B

**First**. I cropped and cut the images into 256x256 tiles, scaling them from 0 to 1 for future math. To handle this efficiently without bottlenecks, I built a high-performance pipeline using the `tf.data` API to prefetch batches directly into memory. Here is the dataset result:

images: 1305
masks: 1305
images dataset shape: (1305, 256, 256, 3)
masks dataset shape: (1305, 256, 256, 3)

**Second**. I converted labels from hex to RGB for future model analysis. Here are the RGB results for each class:

1. Building: [60, 16, 152]
2. Land:     [132, 41, 246]
3. Road:     [110, 193, 228]
4. Veg:      [254, 221, 58]
5. Water:    [226, 169, 41]
6. Unlabeled:[155, 155, 155]

**Third**. I created "label images", meaning an empty canvas where I placed integers from 0 to 5 to represent each class. I did this because the U-NET model outputs a probability vector for each pixel. For example: `[0.05, 0.85, 0.02, 0.01, 0.02, 0.05]`. In this case, class 1 has the highest probability (85%). During training, our evaluation metrics go through these predictions and the ground truth to compare results and update the weights.
I also added an additional channel dimension because Keras and TensorFlow require a `(Batch Size, Height, Width, Channels)` structure. Adding that extra dimension at `axis=3` changed the shape to `(Num_Images, 256, 256, 1)`.

**Fourth**. Now I have a label structure of `(256, 256, 1)`. However, the U-NET model outputs an image as `(256, 256, 6)` to match our 6-class structure (where channel 0 could be water, channel 3 buildings, etc.). To calculate the loss, we need to subtract the prediction from the ground truth, but you cannot mathematically subtract a 6-channel tensor from a 1-channel tensor.

Additionally, the model can treat an array of integers `[0, 1, 2, 3, 4, 5]` incorrectly as ordinal data. For example, it might think that class 3 (Buildings) is greater than class 0 (Water), or that mistaking a Building (3) for a Road (2) is a "better" mistake than mistaking it for Water (0) because the numbers are closer. One-hot encoding fixes this by converting these classes into independent dimensions:

* Class 1 becomes: `[0, 1, 0, 0, 0, 0]`
* Class 5 becomes: `[0, 0, 0, 0, 0, 1]`

After conversion, I split the data into a classic 80% train / 20% test split. To artificially expand our dataset size and teach the model to be invariant to orientation, I integrated rigid geometric augmentations (90-degree rotations and horizontal/vertical flips) which preserve the true scale of satellite imagery.

**Fifth**. I implemented the Jaccard Coefficient, also known as Intersection over Union (IoU). While pixel accuracy only checks how many individual pixels you got right, the Jaccard Coefficient measures how well the predicted shapes actually overlap with the true shapes. Our U-NET model predicts boundaries in the image; we take those predictions, unite them with the ground truth, and check where they intersect.

**Sixth**. I created a custom architecture of encoder (downsampling) and decoder (upsampling) layers for the U-NET model. When looking at satellite images, macro-features like a whole lake or land area are visible, but we need deep features to classify them exactly. The encoder compresses the data—think of it like a "squint" to find global context and understand exactly *what* is in the image. We apply this operation several times, sliding more kernels across the image as it compresses. Once done, the model understands the global context, but the spatial accuracy is compromised, making the image look like a giant blocky square. The decoder fixes this by upsampling the details back (from 16x16 to 32x32, etc.) until we reach the original size, using skip connections to recover the exact *where* of the boundaries.

**Seventh**. I defined evaluation metrics and loss functions. The first two metrics—Accuracy and Jaccard Coefficient—are for human evaluation.

1. **Accuracy** is a simple metric that checks each predicted pixel against the ground truth. It is not the best metric for an imbalanced dataset because if land dominates, the model can predict land everywhere to achieve high accuracy while failing completely on rare classes.
2. **Jaccard Coefficient** is used to monitor true IoU.

For the model's optimization, I combined two different loss functions into a single total loss:

1. **Dice Loss (shape punisher)** measures the overlap between the predicted shape and the ground truth, punishing the model heavily if the structural boundaries are off.
2. **Categorical Focal Loss (hard class spotlight)** applies a mathematical modifier that dynamically scales down the loss of easy, highly-confident pixels. If the model is 99% sure a pixel is land, Focal Loss reduces its importance to almost zero. If the model is confused by a tricky road pixel, Focal Loss magnifies that error. This forces the U-Net to stop wasting energy on what it already knows and focus entirely on learning hard features like road lines and building edges.

**Final**. We use the Adam optimizer to adjust weights during training. To accelerate this process, I configured the entire pipeline to run across dual GPUs using a mirrored distribution strategy, splitting the workload synchronously.

Adam uses a learning rate to define how aggressively it changes weight values. I set up a learning rate scheduler that monitors validation loss; if it stops improving for 5 epochs, it cuts the learning rate in half, helping the model make smaller adjustments to map tricky details. I also configured Early Stopping to monitor validation loss. If it fails to improve for 12 epochs (allowing for 2 learning rate cuts), the model will begin overfitting and memorizing pixels. The training automatically stops here and restores the best historical weights. Finally, I started the 100-epoch training run.