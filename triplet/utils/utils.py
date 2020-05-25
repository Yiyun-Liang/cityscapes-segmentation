import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
import shutil
from PIL import Image
from random import randint, sample

from dataset.dataloader import CityscapesVideos, TripletCityscapesVideos

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
       transforms.Resize((128,256)),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
       transforms.Resize((128,256)),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def save_images(outputs, batch_idx, out_dir):
    #unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    file_IDs = ['pred','real','previous']
    for out, file_id in zip(outputs, file_IDs):
        inputs = torch.mul(out, 255.0)
        img = np.uint8(np.clip(inputs.cpu().squeeze(0).transpose(0,2).transpose(0,1).squeeze(2).detach().numpy(), 0, 255))
        img = Image.fromarray(img)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        img.save('{}/{}_{}.jpg'.format(out_dir, batch_idx, file_id))

def get_dataset(train_dir,  test_dir, frames):
    transform_train, transform_test = get_transforms()
    trainset = TripletCityscapesVideos(train_dir, transform_train, frames)
    testset = TripletCityscapesVideos(test_dir, transform_test, frames, test=True)

    return trainset, testset
