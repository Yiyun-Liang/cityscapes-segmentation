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
        # self.data_info = pd.read_csv(csv_path, header=0, nrows=n_rows)
        self.data_info = pd.read_csv(csv_path, header=0)
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
        #prev_id = np.random.randint(low=start_frame, high=current_id+1)
        #last_id = np.random.randint(low=current_id, high=end_frame+1)

        # Open the images
        first_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame).zfill(6))
        first_img = Image.open(first_image_name)

        second_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+4).zfill(6))
        second_img = Image.open(second_image_name)

        third_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+8).zfill(6))
        third_img = Image.open(third_image_name)

        last_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+12).zfill(6))
        last_img = Image.open(last_image_name)

        # Transform the images
        first_grey = np.asarray(first_img.convert("L"))
        second_grey = np.asarray(second_img.convert("L"))
        third_grey = np.asarray(third_img.convert("L"))
        last_grey = np.asarray(last_img.convert("L"))
        first_grey = np.concatenate((first_grey[:, :, None], first_grey[:, :, None], first_grey[:, :, None]), axis=2)
        second_grey = np.concatenate((second_grey[:, :, None], second_grey[:, :, None], second_grey[:, :, None]), axis=2)
        third_grey = np.concatenate((third_grey[:, :, None], third_grey[:, :, None], third_grey[:, :, None]), axis=2)
        last_grey = np.concatenate((last_grey[:, :, None], last_grey[:, :, None], last_grey[:, :, None]), axis=2)
        first_grey = self.transforms(Image.fromarray(first_grey))

        second_grey = self.transforms(Image.fromarray(second_grey))
        third_grey = self.transforms(Image.fromarray(third_grey))
        last_grey = self.transforms(Image.fromarray(last_grey))
        first_img = self.transforms(first_img)
        second_img = self.transforms(second_img)
        third_img = self.transforms(third_img)
        last_img = self.transforms(last_img)

        # Make images temporal pairs, shuffle, and make label
        pair = {}
        pair[first_img] = 0
        pair[second_img] = 1
        pair[third_img] = 2
        pair[last_img] = 3
        img_list = [first_grey, second_grey, third_grey, last_grey]
        label_dict = {'012':0, '021':1, '102':2, '120':3, '201':4, '210':5}
        # random.shuffle(img_list)
        # label = str(pair[img_list[0]]) + str(pair[img_list[1]]) + str(pair[img_list[2]], str(pair[img_list[3]]))
        # img_list_label = label_dict[label]
        label = [first_img, second_img, third_img, last_img]
        #img_list_label = pair[img_list[1]]

        return (img_list, label)

    def __len__(self):
        return self.data_len
