"""
How to Run on CityScapes Dataset:
    python3.6 train.py
    --train_dir /atlas/u/buzkent/MMVideoPredictor/data/CityScapes/debug.csv
    --test_dir /atlas/u/buzkent/MMVideoPredictor/data/CityScapes/debug.csv
    --frames 0 3 6 9 12
    --satellite (Only if you want to use satellite images)
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
from models.net import TemporalNet
from utils import utils
#from pytorch_msssim import ssim

import torchvision.models as models
import argparse

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='VideoPredictor Training')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--data_dir', default='data/', help= 'data directory')
parser.add_argument('--train_dir', default='data/', help='training data directory')
parser.add_argument('--test_dir', default='data/', help='test data directory')
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

def train(epoch):
    temporal_net.train()
    #l1, ssim_loss = [], []
    matches, losses = [], []
    for batch_idx, (inputs, label) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        frame1, frame2, frame3 = inputs[0], inputs[1], inputs[2]

        if not args.parallel:
            frame1 = frame1.to(device)
            frame2 = frame2.to(device)
            frame3 = frame3.to(device)
            label = label.to(device)

        predicted_label = temporal_net(frame1, frame2, frame3)
        loss = criterion(predicted_label, label)

        #l1.append(criterion1.cpu())
        #ssim_loss.append(criterion2.detach().cpu())
        losses.append(loss.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print('target label: {}, predicted_label: {}'.format(label.cpu(), predicted_label.cpu()))

    loss = torch.stack(losses).mean()
    #l1_loss = torch.stack(l1).mean()
    #ssim_loss = torch.stack(ssim_loss).mean()
    log_value('Train Total Loss', loss, epoch)
    #log_value('Train L1 Loss', l1_loss, epoch)
    #log_value('Train SSIM Loss', ssim_loss, epoch)
    log_str = 'Train Epoch: %d | Loss: %.3f'%(epoch, loss)
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    print(log_str)

def test(epoch):
    global best_loss
    temporal_net.eval()
    #l1, ssim_loss = [], []
    matches, losses = [], []
    with torch.no_grad():
        for batch_idx, (inputs, label) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            frame1, frame2, frame3 = inputs[0], inputs[1], inputs[2]

            if not args.parallel:
                frame1 = frame1.to(device)
                frame2 = frame2.to(device)
                frame3 = frame3.to(device)
                label = label.to(device)

            predicted_label = temporal_net(frame1, frame2, frame3)
            loss = criterion(predicted_label, label)

            losses.append(loss.detach().cpu())

    loss = torch.stack(losses).mean()
    #l1_loss = torch.stack(l1).mean()
    #ssim_loss = torch.stack(ssim_loss).mean()

    log_value('Validation Total Loss', loss, epoch)
    #log_value('Validation L1 Loss', l1_loss, epoch)
    #log_value('Validation MSSSIM Loss', ssim_loss, epoch)
    log_str = 'Test Epoch: %d | Loss: %.3f'%(epoch, loss)
    writer.add_scalar('Loss/val', loss.item(), epoch)

    print(log_str)
    rnet_state_dict = temporal_net.module.state_dict() if args.parallel else temporal_net.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d.pth'%(epoch))
    if loss < best_loss:
        torch.save(rnet_state_dict, args.cv_dir+'/best_loss.pth')
        best_loss = loss

def adjust_learning_rate(epoch, args):
    """Decay the learning rate based on schedule"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch // 10))
    
# Main Function Start Here
writer = SummaryWriter()

trainset, testset = utils.get_dataset(args.train_dir, args.test_dir, args.frames)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.num_workers)

c = torch.cuda.device_count()
print('Number of GPUs:', c)
if c > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

rnet = models.resnet101(pretrained=True).to(device)
# Remove the last layer and extract the maxpooling features
del rnet.fc
rnet.fc=lambda x:x
temporal_net = TemporalNet(rnet, 2048, 3).to(device)

start_epoch = 0
best_loss = 1000
#if args.ckpt_dir:
#    ckpt = torch.load(args.ckpt_dir)
#    rnet.load_state_dict(ckpt['rnet'])
#    start_epoch = int(args.ckpt_dir.split('_')[-1])

#if c > 1:
#    rnet = nn.DataParallel(rnet, device_ids=[0, 1, 2, 3])
#device = torch.device("cuda")
#rnet.to(device)

# Save the configuration to the output directory
configure(args.cv_dir+'/log', flush_secs=5)
optimizer = optim.SGD(temporal_net.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    print('Start training epoch {}'.format(epoch))
    adjust_learning_rate(epoch, args)
    train(epoch)
    if epoch % 5 == 0:
        test(epoch)
writer.close()
