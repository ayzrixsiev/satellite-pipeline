# Version 2

**First**. This time, instead of using custom U-NET layers, i set up pre-trained backbone architecture resnet34 and imagenet weights, so that model can focus on identifying classes, and skip learning basics of different shapes.

**Second**. I resolved issues with creating label images, integer values were mapped incorrectly when label images were created.

**Third**. I noticed that the model is training on one GPU instead of using, both given by kaggle, so i fixed that as well.

**Fourth**. One of the biggest changes i made was using different weight values for each class. Because aerial images usually suffer a lot of imbalance. I improved Jaccard Coefficient, and added checkpoint function, which watches IoU each epoch, and if there is a change to the better from previous iteration, then it saves the weights into the model.keras weights.