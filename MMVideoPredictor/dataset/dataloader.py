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
    def __init__(self, csv_path, transform, frame_idxs, transform_emb, embeddings, num_rows=None, test=False):
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
        self.transform_emb = transform_emb
        self.use_emb = embeddings
        # Read the csv file
        if num_rows is None:
            self.data_info = pd.read_csv(csv_path, header=0)
        else:
            self.data_info = pd.read_csv(csv_path, header=0, nrows=num_rows)
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
        # emb_as_tensor = []
        # img_as_tensor = []

        start_frame = self.start_frames[index]
        end_frame = self.end_frames[index]
        annotated_frame = self.annotated_frames[index]


        first_input_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame).zfill(6))
        first_input_img = Image.open(first_input_image_name)

        second_input_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+2).zfill(6))
        second_input_img = Image.open(second_input_image_name)

        third_input_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+4).zfill(6))
        third_input_img = Image.open(third_input_image_name)


        first_output_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+6).zfill(6))
        first_output_img = Image.open(first_output_image_name)

        second_output_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+8).zfill(6))
        second_output_img = Image.open(second_output_image_name)

        third_output_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, str(seq_id).zfill(6), str(start_frame+10).zfill(6))
        third_output_img = Image.open(third_output_image_name)


        first_input_img = self.transforms(first_input_img)[:, None, ...]
        second_input_img = self.transforms(second_input_img)[:, None, ...]
        third_input_img = self.transforms(third_input_img)[:, None, ...]


        first_output_img = self.transforms(first_output_img)[:, None, ...]
        second_output_img = self.transforms(second_output_img)[:, None, ...]
        third_output_img = self.transforms(third_output_img)[:, None, ...]

        img_list = torch.cat((first_input_img, second_input_img, third_input_img), axis=1)

        label = torch.cat((first_output_img, second_output_img, third_output_img), axis=1)

        return (img_list, label)







        # if self.use_emb:
        #     for idx in self.frame_idxs:
        #         # load emb from file
        #         emb_as_tensor.append(embedding)

        # for idx in self.frame_idxs:
        #     # Open image
        #     # eg. <sequence_name>/aachen_000000_000000_leftImg8bit.png
        #     single_image_name = '{}/{}_{}_{}_leftImg8bit.png'.format(sequence_name, folder_name, seq_id.zfill(6), str(start_frame+int(idx)).zfill(6))
        #     img_as_img = Image.open(single_image_name) #.convert('L')
        #     # Transform the image
        #     img_as_tensor.append(self.transforms(img_as_img))

        # return torch.stack(emb_as_tensor, dim=0), torch.stack(img_as_tensor, dim=0)

    def __len__(self):
        return self.data_len