"""
How to Run on CityScapes Dataset:
    python3.6 test.py
    --test_dir ./data/CityScapes/debug.csv
    --output_dir ./cv/output/
    --ckpt_dir ./cv/debug/ckpt_E_130
    --batch_size 1
    --start_frame 0
    --end_frame 5
    --sampling_rate 1
"""
import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import pdb
from models import PredictorNet
from utils import utils
from pytorch_msssim import ssim

import argparse
parser = argparse.ArgumentParser(description='VideoPredictor Testing')
parser.add_argument('--test_dir', default='data/', help='test data directory')
parser.add_argument('--output_dir', default='cv/tmp/', help='output directory for the saved images')
parser.add_argument('--ckpt_dir', default='cv/tmp/', help='Model Checkpoint')
parser.add_argument('--frames', nargs='+', help='Frames to use as input and output', required=True)
parser.add_argument('--satellite', action='store_true', help='Incorporates Satellite Image')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers in training and testing')
args = parser.parse_args()

def test(epoch):

    rnet.eval()
    matches, losses = [], []
    for batch_idx, inputs in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs = inputs.cuda()

        targets = inputs[:, inputs.shape[1]-1, :, :, :]
        previous = inputs[:, inputs.shape[1]-2, :, :, :]
        inputs = inputs[:, :inputs.shape[1]-1, :, :, :]
        preds = rnet.forward(inputs, args.satellite)

        loss = torch.mean(torch.abs(preds-targets))
        loss_ssim = ssim(torch.clamp(preds, min=0, max=1), torch.clamp(targets, min=0, max=1), data_range=1, size_average=True)
        losses.append(loss_ssim.cpu().detach())

        # utils.save_images([preds, targets, previous], batch_idx, args.output_dir)

    loss = torch.stack(losses).mean()
    log_str = 'Loss: %.3f'%(loss)
    print(log_str)


_, testset = utils.get_dataset(args.test_dir, args.test_dir, args.frames, args.satellite)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
rnet = PredictorNet.SpatioTemporalNet(len(args.frames)-1)
checkpoint = torch.load(args.ckpt_dir)
rnet.load_state_dict(checkpoint['rnet'])
rnet.cuda()

test(0)
