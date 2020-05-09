import pandas as pd
import numpy as np
import warnings
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name).convert('RGB')
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromVideos(Dataset):
    def __init__(self, csv_path, transform, transform_sat, frame_idxs, satellite):
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
        self.transforms_sat = transform_sat
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.seq_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.gt_seq_idx = 19
        # Video details
        self.frame_idxs = frame_idxs
        # Modality
        self.satellite = satellite
        self.path_to_satellite = '/atlas/u/buzkent/CityScapes/GoogleMaps/'

    def __getitem__(self, index):
        # Get image name from the pandas df
        sequence_name = self.image_arr[index]
        first_frame_idx = int(self.seq_arr[index]) - self.gt_seq_idx
        img_as_tensor = []

        img_as_img = Image.open('{}{}_{}.jpg'.format(self.path_to_satellite, sequence_name.split('/')[-1],
            str(self.seq_arr[index]).zfill(6))) #.convert('L')
        img_as_tensor.append(self.transforms_sat(img_as_img))

        for idx in self.frame_idxs:
            # Open image
            single_image_name = '{}_{}_leftImg8bit.png'.format(sequence_name, str(first_frame_idx+int(idx)).zfill(6))
            img_as_img = Image.open(single_image_name) #.convert('L')
            # Transform the image
            img_as_tensor.append(self.transforms(img_as_img))

        return torch.stack(img_as_tensor, dim=0)

    def __len__(self):
        return self.data_len
