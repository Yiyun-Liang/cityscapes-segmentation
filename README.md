# cityscapes-segmentation


This repo explores different self-supervised pretext task for semantic segmentation on cityscapes dataset (original resolution 1024x2048).

### Pretext Tasks
Pretext tasks we are implementing(using 15,000 frames of video data from cityscapes dataset):

* triplet loss on frame temporal location in `/triplet`
* frame order prediction
* video colorization

### Target Tasks
Our target tasks include semantic segmentation and future frame prediction.

* For the target task of semantic segmentation, we evaluate a UNet and a Deeplabv3 model on the cityscapes dataset(5000 examples with fine annotations).
* For the task of future frame prediction, we evaluate an encoder-decoder temporal network on the cityscapes dataset.

### Code Structure

* `/unet`: code for running UNet for semantic segmentation (work based on [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet))
* `/DeepLabv3`: code for running DeepLabv3 for semantic segmentation (work based on [DeepLabv3.pytorch](https://github.com/chenxi116/DeepLabv3.pytorch))
* `/triplet`: pretext task to generate embeddings for video frames using triplet loss (work based on [uzkent/MMVideoPredictor](https://github.com/uzkent/MMVideoPredictor))
* `/spatioTemporal`: pretext task doing video frame order prediction (work based on [uzkent/MMVideoPredictor](https://github.com/uzkent/MMVideoPredictor))
* `/colorization`: pretext task doing video frame colorization 
* `/MMVideoPredictor`: future frame generation using custom temporal network (work based on [uzkent/MMVideoPredictor](https://github.com/uzkent/MMVideoPredictor))

### Preliminary Experiments

* UNet in `/unet` trained from scratch 
* Deeplabv3 model in `/DeepLabv3` trained from scratch 
* Deeplabv3 model finetuned on pretrained imagenet weights
* Deeplabv3 model finetuned on pretrained weights from our pretext tasks

Model  | Setup | mIoU (acc)
------------- | ------------- | ---
**UNet** (lr=0.001, ReduceLROnPlateau, RMSprop(weight_decay=1e-8, momentum=0.9), CrossEntropyLoss) | downsample 2x, bs=1, 30 epochs | 0.5153 (0.8653)
**UNet**(...) | downsample 4x, bs=8, 30 epochs | 0.4613 (0.85)
**UNet**(...)  | downsample 8x, bs=64, 30 epochs | 0.45 (0.85)
**DeepLabv3**(resnet101) | ImagetNet pretrained, 10 epochs | 59.31
**DeepLabv3**(resnet101) | scratch, 10 epochs | 27.24
**DeepLabv3**(resnet101) | scratch, 100 epochs | 49.79