import pandas as pd
import numpy as np
import warnings
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CityscapesVideos(Dataset):
    def __init__(self, csv_path, transform, frame_idxs):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
            start_frame : Start frame of the video sequences
            end_frame : End frame of the video sequences
            sampling_rate : Sampling rate of video frames
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.seq_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.gt_seq_idx = 19
        # Video frames needed
        self.frame_idxs = frame_idxs

    def __getitem__(self, index):
        # Get image name from the pandas df
        sequence_name = self.image_arr[index]
        folder_name = sequence_name.split('/')[-1]
        seq_id = str(self.seq_arr[index])
        # first_frame_idx = int(self.seq_arr[index]) - self.gt_seq_idx
        img_as_tensor = []

        for idx in self.frame_idxs:
            # Open image
            # eg. <sequence_name>/aachen_000000_000000_leftImg8bit.png
            single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(idx).zfill(6))
            img_as_img = Image.open(single_image_name) #.convert('L')
            # Transform the image
            img_as_tensor.append(self.transforms(img_as_img))

        return torch.stack(img_as_tensor, dim=0)

    def __len__(self):
        return self.data_len


class TripletCityscapesVideos(Dataset):
    def __init__(self, csv_path, transform, frame_idxs):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
            start_frame : Start frame of the video sequences
            end_frame : End frame of the video sequences
            sampling_rate : Sampling rate of video frames
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['path'])
        self.seq_arr = np.asarray(self.data_info['sequence'])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.gt_seq_idx = 19
        # Video frames needed
        self.frame_idxs = frame_idxs

    def __getitem__(self, index):
        # Get image name from the pandas df
        sequence_name = self.image_arr[index]
        folder_name = sequence_name.split('/')[-1]
        seq_id = str(self.seq_arr[index])
        # first_frame_idx = int(self.seq_arr[index]) - self.gt_seq_idx
        img_as_tensor = []

        anchor = np.random.randint(low=0, high=30)
        single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(anchor).zfill(6))
        img_as_img = Image.open(single_image_name) #.convert('L')
        # Transform the image
        anchor_img = self.transforms(img_as_img)

        pos_range = np.concatenate((np.arange(np.maximum(anchor-3, 0), anchor-1), \
                                    np.arange(anchor+1, np.minimum(anchor+3, 29))))
        positive = np.random.choice(pos_range)
        single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(positive).zfill(6))
        img_as_img = Image.open(single_image_name) #.convert('L')
        pos = self.transforms(img_as_img)

        neg_range = np.concatenate((np.arange(0, anchor-10), \
                                    np.arange(anchor+10, 29)))
        negative = np.random.choice(neg_range)
        single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(negative).zfill(6))
        img_as_img = Image.open(single_image_name) #.convert('L')
        neg = self.transforms(img_as_img)

        return (anchor_img, pos, neg)

    def __len__(self):
        return self.data_len