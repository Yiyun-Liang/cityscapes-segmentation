# code based on https://github.com/uzkent/MMVideoPredictor

import pandas as pd
import numpy as np
import warnings
import torch
import random

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class TemporalVideoDataset(Dataset):
    def __init__(self, csv_path, transform, frame_idxs, n_rows):
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
        self.data_info = pd.read_csv(csv_path, header=0, nrows=n_rows)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['path'])
        self.seq_arr = np.asarray(self.data_info['sequence'])
        #self.current_arr = np.asarray(self.data_info['current'])
        self.start_frames = np.asarray(self.data_info['start'])
        self.end_frames = np.asarray(self.data_info['end'])
        self.annotated_frames = np.asarray(self.data_info['annotation'])
        # Calculate len
        self.data_len = len(self.data_info.index)
        # Video frames needed
        self.frame_idxs = frame_idxs

    def __getitem__(self, index):
        # Get image name from the pandas df
        sequence_name = self.image_arr[index]
        folder_name = sequence_name.split('/')[-1]
        seq_id = self.seq_arr[index]
        #current_id = self.current_arr[index]

        start_frame = self.start_frames[index]
        end_frame = self.end_frames[index]
        annotated_frame = self.annotated_frames[index]

        # Randomly pick the triplet with sequence
        first_id = np.random.randint(low=start_frame, high=start_frame+8)
        second_id = np.random.randint(low=start_frame+10, high=start_frame+18)
        third_id = np.random.randint(low=start_frame+20, high=start_frame+28)

        # Open the images
        first_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+8).zfill(6))
        first_img = Image.open(first_image_name)

        second_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+10).zfill(6))
        second_img = Image.open(second_image_name)

        last_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+12).zfill(6))
        last_img = Image.open(last_image_name)

        # Transform the images
        first_img = self.transforms(first_img)
        second_img = self.transforms(second_img)
        last_img = self.transforms(last_img)

        # Make images temporal pairs, shuffle, and make label
        pair = {}
        pair[first_img] = 0
        pair[second_img] = 1
        pair[last_img] = 2
        img_list = [first_img, second_img, last_img]
        label_dict = {'012':0, '021':1, '102':2, '120':3, '201':4, '210':5}
        random.shuffle(img_list)
        label = str(pair[img_list[0]]) + str(pair[img_list[1]]) + str(pair[img_list[2]])
        #print('1', label)
        img_list_label = label_dict[label]
        #print('2', img_list_label)
        #img_list_label = pair[img_list[1]]

        return (img_list, img_list_label)

    def __len__(self):
        return self.data_len
