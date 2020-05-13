# cityscapes-segmentation


This repo explores different self-supervised pretext task for semantic segmentation on cityscapes dataset (original resolution 1024x2048).

Pretext tasks we are implementing(using 15,000 frames of video data from cityscapes dataset):

* triplet loss on frame temporal location in `/triplet`
* frame order prediction
* video colorization

For the target task of semantic segmentation, we evaluate a UNet and a Deeplabv3 model on the cityscapes dataset(5000 examples with fine annotations).

Experiments:

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