from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
import torchvision.transforms as transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, split='train', scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        # mask eg. erfurt_000000_000019_gtFine_color.png
        self.ids = [fol+'/'+splitext(file)[0].split('_leftImg8bit')[0] for fol in listdir(imgs_dir) for file in listdir(imgs_dir + fol) 
                    if file.endswith('_leftImg8bit.png')]

        self.n_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23,24,25,26,27,28,31,32,33]
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
        self.mean = [0.28689554, 0.32513303, 0.28389177]
        self.std = [0.18696375, 0.19017339, 0.18720214]
        self.img_transform = transforms.Compose([
            transforms.Resize((128,256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,256), interpolation=Image.NEAREST),
        ])
        self.augmentation = None
        self.split = split

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        if self.split is not 'test':
            mask_file = glob(self.masks_dir + idx + '_gtFine_labelIds' + '*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        if self.split is not 'test':
            assert len(mask_file) == 1, \
                f'Either no mask or multiple masks found for the ID {idx}: {mask_file} len: {len(mask_file)} dir {self.masks_dir+idx}'
        
        img = Image.open(img_file[0]).convert('RGB')
        if self.split is not 'test':
            mask = Image.open(mask_file[0])
        # Transform
        img_as_tensor = self.img_transform(img)
        if self.split is not 'test':
            mask = self.encode_segmap(np.array(mask, dtype=np.uint8))
            mask_as_tensor = self.mask_transform(mask)
            mask = np.array(mask_as_tensor)

            # ohe_labels = np.zeros((self.n_classes,) + mask.shape[:2])
            # # print(ohe_labels.shape)
            # for c in range(self.n_classes):
            #     ys, xs = np.where(mask == c)
            #     ohe_labels[c, ys, xs] = 1
            # ohe_labels.astype(int)
            # print(ohe_labels.shape)
            return {'image': img_as_tensor, 'mask': torch.from_numpy(mask).long()}
        else:
            return {'image': img_as_tensor}

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
