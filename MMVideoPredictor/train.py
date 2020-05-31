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
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as torch_models

import argparse
parser = argparse.ArgumentParser(description='VideoPredictor Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_dir', default='data/', help= 'data directory')
parser.add_argument('--train_dir', default='data/', help='training data directory')
parser.add_argument('--test_dir', default='data/', help='test data directory')
parser.add_argument('--frames', nargs='+', help='Frames to use as input and output')
parser.add_argument('--cv_dir', default='cv/from_scratch/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--ckpt_dir', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in training and testing')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--embeddings', action='store_true', default=False, help='incorporates learned embeddings')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    rnet.train()
    l1, ssim_loss = [], []
    matches, losses = [], []
    for batch_idx, inputs in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        # emb, images = inputs
        # if not args.parallel:
        #     emb = emb.cuda()
        #     images = images.cuda()
        input_img, targets = inputs

        if not args.parallel:
            input_img = input_img.cuda()
            targets = targets.cuda()



        # # last three frames as target
        # targets = torch.tanh(torch.cat(( \
        #     images[:, images.shape[1]-3, :, :, :], \
        #     images[:, images.shape[1]-2, :, :, :], \
        #     images[:, images.shape[1]-1, :, :, :]), dim=1))
        # # all other frames + embedding(optional) as input
        # images = images[:, :images.shape[1]-3, :, :, :]
        # v_inputs = images.data


        # preds = torch.tanh(rnet.forward(images, emb, args.embeddings))

        preds = torch.tanh(rnet.forward(input_img))
        preds = preds.reshape(-1, preds.shape[2], preds.shape[3], preds.shape[4])
        targets = targets.reshape(-1, targets.shape[2], targets.shape[3], targets.shape[4])



        criterion1 = torch.mean(torch.abs(preds - targets))
        criterion2 = ssim(preds, targets, val_range=targets.max()-targets.min())
        loss = criterion1

        l1.append(criterion1.cpu())
        ssim_loss.append(criterion2.detach().cpu())
        losses.append(loss.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = torch.stack(losses).mean()
    l1_loss = torch.stack(l1).mean()
    ssim_loss = torch.stack(ssim_loss).mean()

    log_value('Train Total Loss', loss, epoch)
    log_value('Train L1 Loss', l1_loss, epoch)
    log_value('Train SSIM Loss', ssim_loss, epoch)

    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('L1_loss/train', l1_loss, epoch)
    writer.add_scalar('SSIM_loss/train', ssim_loss, epoch)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    log_str = 'Train Epoch: %d | Loss: %.3f | L1: %.3f | MS-SSIM: %.4f'%(epoch, loss, l1_loss, ssim_loss)
    # log_str = 'Train Epoch: %d | Loss: %.3f | L1: %.3f'%(epoch, loss, l1_loss)
    print(log_str)

def test(epoch):
    global best_loss
    rnet.eval()
    l1, ssim_loss = [], []
    matches, losses = [], []
    for batch_idx, inputs in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        # emb, images = inputs
        # if not args.parallel:
        #     emb = emb.cuda()
        #     images = images.cuda()

        # # last three frames as target
        # targets = torch.tanh(torch.cat(( \
        #     images[:, images.shape[1]-3, :, :, :], \
        #     images[:, images.shape[1]-2, :, :, :], \
        #     images[:, images.shape[1]-1, :, :, :]), dim=1))
        # # all other frames + embedding(optional) as input
        # images = images[:, :images.shape[1]-3, :, :, :]

        # preds = torch.tanh(rnet.forward(images, emb, args.embeddings))
        input_img, targets = inputs

        if not args.parallel:
            input_img = input_img.cuda()
            targets = targets.cuda()

        preds = torch.tanh(rnet.forward(input_img))
        preds = preds.reshape(-1, preds.shape[2], preds.shape[3], preds.shape[4])
        targets = targets.reshape(-1, targets.shape[2], targets.shape[3], targets.shape[4])

        criterion1 = torch.mean(torch.abs(preds - targets))
        criterion2 = ssim(preds, targets, val_range=targets.max()-targets.min())
        loss = criterion1

        l1.append(criterion1.detach().cpu())
        ssim_loss.append(criterion2.detach().cpu())
        losses.append(loss.detach().cpu())

    loss = torch.stack(losses).mean()
    l1_loss = torch.stack(l1).mean()
    ssim_loss = torch.stack(ssim_loss).mean()

    log_value('Validation Total Loss', loss, epoch)
    log_value('Validation L1 Loss', l1_loss, epoch)
    log_value('Validation MSSSIM Loss', ssim_loss, epoch)
    writer.add_scalar('Loss/val', loss, epoch)
    writer.add_scalar('L1_loss/val', l1_loss, epoch)
    writer.add_scalar('SSIM_loss/val', ssim_loss, epoch)

    log_str = 'Test Epoch: %d | Loss: %.3f | L1: %.3f | MS-SSIM: %.4f'%(epoch, loss, l1_loss, ssim_loss)
    # log_str = 'Test Epoch: %d | Loss: %.3f | L1: %.3f'%(epoch, loss, l1_loss)
    print(log_str)
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d.pth'%(epoch))
    if loss < best_loss:
        torch.save(rnet_state_dict, args.cv_dir+'/best_loss.pth')
        best_loss = loss

writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.batch_size}')

trainset, testset = utils.get_dataset(args.train_dir, args.test_dir, args.frames, args.embeddings)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.num_workers)
# rnet = PredictorNet.SpatioTemporalNet(len(args.frames)-3, 3)
# if pretrained:
#         print('pretrained resnet50 loading...')
#         model_dict = model.state_dict()
#         if custom is not None:
#             print('loading custom ckpt')
#             pretrained_dict = torch.load(custom)
#         else:
#             print('loading imagenet pretrained resnet50')
#             pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
#         overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(overlap_dict)
#         model.load_state_dict(model_dict)
#     return model


# resnet = torch_models.resnet50(pretrained=False)





device = torch.device("cuda")
resnet = torch_models.resnet50(pretrained=False)
resnet = nn.Sequential(*list(resnet.children())[:-2]).to(device)
rnet = PredictorNet.SpatioTemporalNet(3, 3, resnet)

start_epoch = 0
best_loss = 1000
if args.ckpt_dir:
    ckpt = torch.load(args.ckpt_dir)
    rnet.load_state_dict(ckpt['rnet'])
    start_epoch = int(args.ckpt_dir.split('_')[-1])

c = torch.cuda.device_count()
print('Number of GPUs:', c)
# if c > 1:
#     rnet = nn.DataParallel(rnet, device_ids=[0, 1, 2, 3])
rnet.to(device)

# Save the configuration to the output directory
configure(args.cv_dir+'/log', flush_secs=5)
optimizer = optim.Adam(rnet.parameters(), lr=args.lr)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % 5 == 0:
        test(epoch)

writer.close()