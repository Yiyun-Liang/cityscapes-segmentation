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
from models.net import ColorNet
from utils import utils
#from pytorch_msssim import ssim

import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser(description='VideoPredictor Training')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--data_dir', default='data/', help= 'data directory')
parser.add_argument('--train_dir', default='data/', help='training data directory')
parser.add_argument('--test_dir', default='data/', help='test data directory')
parser.add_argument('--frames', nargs='+', help='Frames to use as input and output', required=False)
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--ckpt_dir', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in training and testing')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
args = parser.parse_args()


if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):

    colornet.train()
    #l1, ssim_loss = [], []
    matches, losses = [], []
    for batch_idx, (inputs, label) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        frame1, frame2, frame3, frame4 = inputs


        label = torch.cat((label[0][:, None, :, :, :], label[1][:, None, :, :, :], label[2][:, None, :, :, :], label[3][:, None, :, :, :]), axis=1)
        # print(label.shape)
        #raise
        frames = torch.cat((frame1, frame2, frame3, frame4), axis=0)
        # print(frames.shape)
        # raise
        #frame1 = torch.cat((frame1, frame1, frame1), dim=1)
        #frame2 = torch.cat((frame2, frame2, frame2), dim=1)
        #frame3 = torch.cat((frame3, frame3, frame3), dim=1)
        #frame4 = torch.cat((frame4, frame4, frame4), dim=1)

        if not args.parallel:
            frame1 = frame1.to(device)
            frame2 = frame2.to(device)
            frame3 = frame3.to(device)
            frame4 = frame4.to(device)
            # frames = frames.to(device)

            label = label.to(device)



        # output_1 = colornet(frame1)
        # output_2 = colornet(frame2)
        # output_3 = colornet(frame3)
        # output_4 = colornet(frame4)
        # output = torch.cat((output_1[:, None, ...], output_2[:, None, ...], output_3[:, None, ...], output_4[:, None, ...]), axis=1)
        output = colornet(frame1, frame2, frame3, frame4)
        output = output.transpose(-1, -2)
        output = output.transpose(-1, -3)
        # print(output.shape)
        # raise
        left_term = output[:, :-1, ...]
        right_term = torch.cat((output[:, -1:, ...], output[:, -1:, ...], output[:, -1:, ...]), axis=1)
        term_shape = left_term.shape
        left_term = torch.reshape(left_term, [-1] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])
        right_term = torch.reshape(right_term, [-1] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])

        feature_prod = torch.matmul(left_term.transpose(2, 1), right_term)
        feature_prod = torch.reshape(feature_prod, [-1] + [feature_prod.shape[-2] * feature_prod.shape[-1]])
        feature_prod = torch.cat((feature_prod[..., None], feature_prod[..., None], feature_prod[..., None]), axis=-1)
        # print(left_term.shape)
        # print(right_term.transpose(2, 1).shape)
        # print(feature_prod.shape)
        # feature_prod = nn.Softmax(1)(feature_prod)
        label = label.transpose(-1, -2)
        label = label.transpose(-1, -3)

        ref_colorGT = label[:, :3, ...]
        tar_colorGT = torch.cat((label[:, -1:, ...], label[:, -1:, ...], label[:, -1:, ...]), axis=1)

        ref_colorGT_reshape = torch.reshape(ref_colorGT, [-1] + [ref_colorGT.shape[-3] * ref_colorGT.shape[-2]]
                                         + [ref_colorGT.shape[-1]])
        tar_colorGT_reshape = torch.reshape(tar_colorGT, [-1] + [tar_colorGT.shape[-3] * tar_colorGT.shape[-2]]
                                         + [tar_colorGT.shape[-1]])
        # print(ref_colorGT_reshape.shape, feature_prod.transpose(2, 1).shape)
        pred_color = ref_colorGT_reshape * feature_prod
        # pred_color = nn.Softmax(-1)(pred_color)

        colorPred = torch.reshape(pred_color, [-1] + [3]
                                     + list(label.shape)[2:])
        
        
        max_cls = torch.max(colorPred, 1)
        # print(colorPred)
        #print(colorPred.shape)
        #print(label[:, 3, ...].shape)
        # print(colorPred[:, -1, ...].shape, label[:, 3, ...].shape)
        loss = criterion(colorPred + 1e-9, tar_colorGT + 1e-9)
        

        #l1.append(criterion1.cpu())
        #ssim_loss.append(criterion2.detach().cpu())
        #losses.append(loss.cpu())
        # print(loss.cpu())
        if loss.cpu() is None or loss.detach().cpu().numpy() == np.nan or np.isnan(loss.detach().cpu().numpy()):
            print("nan here")
        else:
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
    writer.add_embedding(label[:, 3, ...].reshape(args.batch_size, -1), global_step=epoch, tag='input_label')
    writer.add_embedding(colorPred.reshape(args.batch_size, -1), global_step=epoch, tag='input_predict')
    print(log_str)

def test(epoch):
    global best_loss
    colornet.eval()
    #l1, ssim_loss = [], []
    matches, losses = [], []
    with torch.no_grad():
        for batch_idx, (inputs, label) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            frame1, frame2, frame3, frame4 = inputs


            label = torch.cat((label[0][:, None, :, :, :], label[1][:, None, :, :, :], label[2][:, None, :, :, :], label[3][:, None, :, :, :]), axis=1)

            frames = torch.cat((frame1, frame2, frame3, frame4), axis=0)

            #frame1 = torch.cat((frame1, frame1, frame1), dim=1)
            #frame2 = torch.cat((frame2, frame2, frame2), dim=1)
            #frame3 = torch.cat((frame3, frame3, frame3), dim=1)
            #frame4 = torch.cat((frame4, frame4, frame4), dim=1)

            if not args.parallel:
                frame1 = frame1.to(device)
                frame2 = frame2.to(device)
                frame3 = frame3.to(device)
                frame4 = frame4.to(device)
                # frames = frames.to(device)

                label = label.to(device)



            # output_1 = colornet(frame1)
            # output_2 = colornet(frame2)
            # output_3 = colornet(frame3)
            # output_4 = colornet(frame4)
            # output = torch.cat((output_1[:, None, ...], output_2[:, None, ...], output_3[:, None, ...], output_4[:, None, ...]), axis=1)
            output = colornet(frame1, frame2, frame3, frame4)
            output = output.transpose(-1, -2)
            output = output.transpose(-1, -3)
            # print(output.shape)
            # raise
            left_term = output[:, :-1, ...]
            right_term = torch.cat((output[:, -1:, ...], output[:, -1:, ...], output[:, -1:, ...]), axis=1)
            term_shape = left_term.shape
            left_term = torch.reshape(left_term, [-1] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])
            right_term = torch.reshape(right_term, [-1] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])

            feature_prod = torch.matmul(left_term.transpose(2, 1), right_term)
            feature_prod = torch.reshape(feature_prod, [-1] + [feature_prod.shape[-2] * feature_prod.shape[-1]])
            feature_prod = torch.cat((feature_prod[..., None], feature_prod[..., None], feature_prod[..., None]), axis=-1)
            label = label.transpose(-1, -2)
            label = label.transpose(-1, -3)

            ref_colorGT = label[:, :3, ...]
            tar_colorGT = torch.cat((label[:, -1:, ...], label[:, -1:, ...], label[:, -1:, ...]), axis=1)

            ref_colorGT_reshape = torch.reshape(ref_colorGT, [-1] + [ref_colorGT.shape[-3] * ref_colorGT.shape[-2]]
                                             + [ref_colorGT.shape[-1]])
            tar_colorGT_reshape = torch.reshape(tar_colorGT, [-1] + [tar_colorGT.shape[-3] * tar_colorGT.shape[-2]]
                                             + [tar_colorGT.shape[-1]])

            #pred_color = torch.matmul(ref_colorGT_reshape, feature_prod.transpose(2, 1))
            # pred_color = nn.Softmax(-1)(pred_color)
            pred_color = ref_colorGT_reshape * feature_prod
            colorPred = torch.reshape(pred_color, [-1] + [3]
                                         + list(label.shape)[2:])
            
            # print(colorPred.shape)
            # print(label[:, 3, ...].shape)            
            max_cls = torch.max(colorPred, 1)



            loss = criterion(colorPred + 1e-9, tar_colorGT + 1e-9)

            #l1.append(criterion1.cpu())
            #ssim_loss.append(criterion2.detach().cpu())
            losses.append(loss.cpu())

    loss = torch.stack(losses).mean()
    #l1_loss = torch.stack(l1).mean()
    #ssim_loss = torch.stack(ssim_loss).mean()

    log_value('Validation Total Loss', loss, epoch)
    #log_value('Validation L1 Loss', l1_loss, epoch)
    #log_value('Validation MSSSIM Loss', ssim_loss, epoch)
    log_str = 'Test Epoch: %d | Loss: %.3f'%(epoch, loss)
    writer.add_scalar('Loss/val', loss.item(), epoch)
    print(log_str)
    rnet_state_dict = colornet.module.state_dict() if args.parallel else colornet.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d.pth'%(epoch))
    if loss < best_loss:
        torch.save(rnet_state_dict, args.cv_dir+'/best_loss.pth')
        best_loss = loss

def adjust_learning_rate(epoch, args):
    """Decay the learning rate based on schedule"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch // 10))



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


rnet = models.resnet50(pretrained=False)
rnet = nn.Sequential(*list(rnet.children())[:-2]).to(device)
# Remove the last layer and extract the maxpooling features
# temporal_net = TemporalNet(rnet, 2048, 3).to(device)
colornet = ColorNet(rnet, 4).to(device)

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
optimizer = optim.SGD(colornet.parameters(), lr=args.lr)
criterion = nn.MSELoss()

for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    print('Start training epoch {}'.format(epoch))
    adjust_learning_rate(epoch, args)
    train(epoch)
    if epoch % 5 == 0:
        test(epoch)
writer.close()
