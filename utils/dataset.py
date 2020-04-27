from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
#from PIL import Image
#import imageio as m
import cv2

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        # mask eg. erfurt_000000_000019_gtFine_color.png
        self.ids = [fol+'/'+splitext(file)[0].split('_leftImg8bit')[0] for fol in listdir(imgs_dir) for file in listdir(imgs_dir + fol) 
                    if file.endswith('_leftImg8bit.png')]

        self.n_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23,24,25,26,27,28,31,32,33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.is_transform = True
        self.augmentation = None
        self.mean = [0.0,0.0,0.0]
        self.img_size = (512, 1024)
        self.img_norm = True

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '_gtFine_labelIds' + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file} len: {len(mask_file)} dir {self.masks_dir+idx}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        img = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_file[0], cv2.IMREAD_UNCHANGED)
        img = np.array(img, dtype=np.uint8)
        mask = self.encode_segmap(np.array(mask, dtype=np.uint8))

        if self.augmentation is not None:
            img, mask = self.augmentation(img, mask)

        if self.is_transform:
            img, mask = self.transform(img, mask)
        
        return {'image': torch.from_numpy(img).float(), 'mask': torch.from_numpy(mask).long()}

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = cv2.resize(img, dsize=(self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, dsize=(self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        # img = torch.from_numpy(img).float()
        # lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
