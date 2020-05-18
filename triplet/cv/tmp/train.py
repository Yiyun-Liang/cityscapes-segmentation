"""
How to Run on CityScapes Dataset:
    python3.6 train.py
    --train_dir video_train_filelist.csv
    --test_dir video_val_filelist.csv
    --frames 0 3 6 9 12
    --cv_dir ./cv/debug/
    --batch_size 8
"""
import os
import time
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import pdb
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from models import PredictorNet
from utils import utils
# from pytorch_msssim import ssim
from utils.losses import TripletLoss
from models.networks import TripletNet, Classification

import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser(description='VideoPredictor Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--data_dir', default='/ssd/leftImg8bit_sequence', help= 'data directory')
parser.add_argument('--train_dir', default='../video_train_filelist.csv', help='training data directory')
parser.add_argument('--test_dir', default='../video_val_filelist.csv', help='test data directory')
parser.add_argument('--frames', nargs='+', help='Frames to use as input and output', required=False)
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--ckpt_dir', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers in training and testing')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch, device):
    triplet_net.train()
    l1, ssim_loss = [], []
    matches, losses = [], []
    for batch_idx, inputs in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):


        # targets = torch.tanh(torch.cat((inputs[:, inputs.shape[1]-3, :, :, :],inputs[:, inputs.shape[1]-2, :, :, :],inputs[:, inputs.shape[1]-1, :, :, :]), dim=1))
        # inputs = inputs[:, :inputs.shape[1]-3, :, :, :]
        # v_inputs = inputs.data
        anchor, pos, neg = inputs

        if not args.parallel:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

        out1, out2, out3 = triplet_net(anchor, pos, neg)
        # print(out1.shape) # BxNxHxW
        # calculate loss over features
        # pick k negative samples from the batch to calculate loss on
        # then calculate grad on them
        loss = triplet_loss(out1, out2, out3)
        # criterion2 = ssim(preds[:,6:9,:,:], targets[:,6:9,:,:], data_range=targets.max()-targets.min())
        # loss = criterion1

        # l1.append(criterion1.cpu())
        # ssim_loss.append(criterion2.detach().cpu())
        losses.append(loss.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = torch.stack(losses).mean()
    # l1_loss = torch.stack(l1).mean()
    # ssim_loss = torch.stack(ssim_loss).mean()
    log_value('Train Total Loss', loss, epoch)
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    # log_value('Train L1 Loss', l1_loss, epoch)
    # log_value('Train SSIM Loss', ssim_loss, epoch)
    log_str = 'Train Epoch: %d | Loss: %.3f '%(epoch, loss)
    print(log_str)

def test(epoch, device):
    global best_loss
    triplet_net.eval()
    l1, ssim_loss = [], []
    matches, losses = [], []
    with torch.no_grad():
        for batch_idx, inputs in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

            anchor, pos, neg = inputs
            if not args.parallel:
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

            out1, out2, out3 = triplet_net(anchor, pos, neg)
            loss = triplet_loss(out1, out2, out3)
            losses.append(loss.cpu())

    loss = torch.stack(losses).mean()

    log_value('Validation Total Loss', loss, epoch)
    log_str = 'Test Epoch: %d | Loss: %.3f '%(epoch, loss)
    writer.add_scalar('Loss/val', loss.item(), epoch)

    print(log_str)
    rnet_state_dict = triplet_net.module.state_dict() if args.parallel else triplet_net.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d.pth'%(epoch))
    if loss < best_loss:
        torch.save(rnet_state_dict, args.cv_dir+'/best_loss.pth')
        best_loss = loss

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    for param_group in optimizer.param_groups:
        # param_group['lr'] = args.lr * (0.1 ** (epoch // 10))
        if epoch < 5:  # warm-up
            if epoch == 0:
                lr = param_group['lr'] * (float(epoch + 1) / 5)
            else:
                lr = param_group['lr'] * (float(epoch + 1) / 5) / (float(epoch) / 5)
        else:
            if epoch // 30 == 0:
                lr = param_group['lr'] * 0.1
            else:
                lr = param_group['lr']
        param_group["lr"] = lr

writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.batch_size}')

trainset, testset = utils.get_dataset(args.train_dir, args.test_dir, args.frames)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.num_workers)

c = torch.cuda.device_count()
print('Number of GPUs:', c)
if c > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# rnet = PredictorNet.SpatioTemporalNet(len(args.frames)-3, 3*3)
rnet = models.resnet101()
rnet = nn.Sequential(*list(rnet.children())[:-1]).to(device)
triplet_net = TripletNet(rnet)

# losses
triplet_loss = TripletLoss(margin=1.0).to(device)

start_epoch = 0
best_loss = 1000
if args.ckpt_dir:
    ckpt = torch.load(args.ckpt_dir)
    triplet_net.load_state_dict(ckpt['triplet_net'])
    start_epoch = int(args.ckpt_dir.split('_')[-1])


if c > 1:
    triplet_net = nn.DataParallel(triplet_net, device_ids=[0, 1, 2, 3])
triplet_net.to(device)

# Save the configuration to the output directory
configure(args.cv_dir+'/log', flush_secs=5)
optimizer = optim.Adam(triplet_net.parameters(), lr=args.lr)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    #adjust_learning_rate(optimizer, epoch, args)
    train(epoch, device)
    if epoch % 5 == 0:
        test(epoch, device)

writer.close()
