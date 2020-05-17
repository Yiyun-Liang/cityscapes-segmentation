# code based on https://github.com/uzkent/MMVideoPredictor

import pandas as pd
import numpy as np
import warnings
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CityscapesVideos(Dataset):
    def __init__(self, csv_path, transform, frame_idxs, test=False):
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
        self.image_arr = np.asarray(self.data_info['path'])
        self.seq_arr = np.asarray(self.data_info['sequence'])
        self.start_frames = np.asarray(self.data_info['start'])
        self.end_frames = np.asarray(self.data_info['end'])
        self.annotated_frames = np.asarray(self.data_info['annotation'])
        # Calculate len
        self.data_len = len(self.data_info.index)
        # Video frames needed
        self.frame_idxs = frame_idxs
        self.is_test = test

    def __getitem__(self, index):
        # Get image name from the pandas df
        sequence_name = self.image_arr[index]
        folder_name = sequence_name.split('/')[-1]
        seq_id = str(self.seq_arr[index])
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
    def __init__(self, csv_path, transform, frame_idxs, test=False):
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
        self.start_frames = np.asarray(self.data_info['start'])
        self.end_frames = np.asarray(self.data_info['end'])
        self.annotated_frames = np.asarray(self.data_info['annotation'])
        # Calculate len
        self.data_len = len(self.data_info.index)
        # Video frames needed
        self.frame_idxs = frame_idxs
        self.is_test = test

    def __getitem__(self, index):
        # Get image name from the pandas df
        sequence_name = self.image_arr[index]
        folder_name = sequence_name.split('/')[-1]
        seq_id = str(self.seq_arr[index])
        img_as_tensor = []

        start_frame = self.start_frames[index]
        end_frame = self.end_frames[index]
        annotated_frame = self.annotated_frames[index]

        if self.is_test:
            anchor = start_frame
        else:
            anchor = np.random.randint(low=start_frame, high=end_frame+1)
        single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(anchor).zfill(6))
        img_as_img = Image.open(single_image_name) #.convert('L')
        # Transform the image
        anchor_img = self.transforms(img_as_img)
        
        if self.is_test:
            positive = start_frame+2
        else:
            pos_range = np.concatenate((np.arange(np.maximum(anchor-3, start_frame), anchor-1), \
                                        np.arange(anchor+1, np.minimum(anchor+3, end_frame))))
            positive = np.random.choice(pos_range)
        single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(positive).zfill(6))
        img_as_img = Image.open(single_image_name) #.convert('L')
        pos = self.transforms(img_as_img)

        if self.is_test:
            negative = start_frame+10
        else:
            neg_range = np.concatenate((np.arange(start_frame, anchor-10), \
                                        np.arange(anchor+10, end_frame)))
            negative = np.random.choice(neg_range)
        single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(negative).zfill(6))
        img_as_img = Image.open(single_image_name) #.convert('L')
        neg = self.transforms(img_as_img)

        return (anchor_img, pos, neg)

    def __len__(self):
        return self.data_len