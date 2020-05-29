import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union, get_ratio, get_centroid
from torch.utils.tensorboard import SummaryWriter

from loss import RatioLoss, CentroidLoss

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet50',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None, 
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--custom', type=str, default=None,
                    help='path to custom ckpt')
parser.add_argument('--crossentropy', action='store_true',
                    help='use crossentropy loss')
parser.add_argument('--ratio', action='store_true',
                    help='use ratio loss')
parser.add_argument('--centroid', action='store_true',
                    help='use centroid loss')
args = parser.parse_args()

best_train_loss = 1000
best_train_miou = 0
best_test_miou = 0

def get_miou(model, dataset, writer, epoch, split):
  torch.cuda.set_device(args.gpu)
  model = model.cuda()
  model.eval()

  inter_meter = AverageMeter()
  union_meter = AverageMeter()
  with torch.no_grad():
    for i in range(len(dataset)):
      inputs, target = dataset[i]
      inputs = Variable(inputs.cuda())
      outputs = model(inputs.unsqueeze(0))
      _, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
      mask = target.numpy().astype(np.uint8)

      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    print('{0} Mean IoU: {1:.2f}'.format(split, iou.mean() * 100))
    writer.add_scalar('miou/{0}'.format(split), iou.mean(), epoch)
    return iou.mean() * 100

def main():
  global best_train_loss, best_train_miou, best_test_miou

  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  path = 'ckpt'
  subpath = ''

  if args.crossentropy:
    subpath = ''.join([subpath, 'crossentropy_'])
  if args.ratio:
    subpath = ''.join([subpath, 'ratio_'])
  if args.centroid:
    subpath = ''.join([subpath, 'centroid_'])

  path = os.path.join(path, subpath, str(args.base_lr))
  if not os.path.exists(path):
    os.makedirs(path)
  model_fname = '{0}/deeplab_{1}_{2}_v3_{3}_epoch%d.pth'.format(
      path, args.backbone, args.dataset, args.exp)
  best_fname = '{0}/deeplab_{1}_{2}_v3_{3}_best.pth'.format(
      path, args.backbone, args.dataset, args.exp)

  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('/ssd',
        train=args.train, crop_size=args.crop_size)

    test_dataset = Cityscapes('/ssd',
        train=False, crop_size=args.crop_size)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        custom=args.custom,
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  elif args.backbone == 'resnet50':
    model = getattr(deeplab, 'resnet50')(
        pretrained=(not args.scratch),
        custom=args.custom,
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

  writer = SummaryWriter(comment=f'LR_{args.base_lr}_BS_{args.batch_size}')
  global_step = 0
  if args.train:
    print('training')
    print('using crossentropy loss {}'.format(args.crossentropy))
    print('using ratio loss {}'.format(args.ratio))
    print('using centroid loss {}'.format(args.centroid))
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    ratio_criterion = RatioLoss()
    centroid_criterion = CentroidLoss()

    model = nn.DataParallel(model).cuda()
    model = model.cuda()
    model.train()
    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters()))
    last_params = list(model.module.aspp.parameters())
    optimizer = optim.SGD([
      {'params': filter(lambda p: p.requires_grad, backbone_params)},
      {'params': filter(lambda p: p.requires_grad, last_params)}],
      lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.train,
        pin_memory=True, num_workers=args.workers)
    max_iter = args.epochs * len(dataset_loader)
    losses = AverageMeter()
    start_epoch = 0

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
      epoch_loss = []
      model.train()
      for i, (inputs, target) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * args.last_mult

        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        outputs = model(inputs)

        if args.crossentropy: 
          loss = criterion(outputs, target)

        if args.ratio:
          ratio_out = get_ratio(outputs).cuda()
          ratio_target = get_ratio(target, target=True).cuda()
          if not args.crossentropy:
            loss = ratio_criterion(ratio_out, ratio_target)
          else:
            loss += ratio_criterion(ratio_out, ratio_target)
        
        if args.centroid:
          c_out = Variable(get_centroid(outputs).cuda())
          c_target = Variable(get_centroid(target, target=True).cuda())
          loss += centroid_criterion(c_out, c_target)

        if np.isnan(loss.item()) or np.isinf(loss.item()):
          pdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        #writer.add_scalar('Loss/train', loss.item(), global_step)
        global_step += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

        epoch_loss.append(loss.cpu())

      loss = torch.stack(epoch_loss).mean()
      writer.add_scalar('Loss/train', loss.item(), epoch)
      writer.add_scalar('learning_rate0', optimizer.param_groups[0]['lr'], epoch)
      writer.add_scalar('learning_rate1', optimizer.param_groups[1]['lr'], epoch)
      train_iou = get_miou(model, dataset, writer, epoch, 'train')
      test_iou = get_miou(model, test_dataset, writer, epoch, 'test')
      if test_iou > best_test_miou:
        best_test_miou = test_iou
        best_train_miou = train_iou
        best_train_loss = loss.item()
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, best_fname)

      if (epoch+1) % 10 == 0:
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, model_fname % (epoch + 1))
      
    print('Best Train Mean IoU: {0:.2f}'.format(best_train_miou))
    print('Best Test Mean IoU: {0:.2f}'.format(best_test_miou))
    print('Best Train Loss: {0:.2f}'.format(best_train_loss))

  else:
    print('testing')
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_fname % args.epochs)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    with torch.no_grad():
      for i in range(len(dataset)):
        inputs, target = dataset[i]
        inputs = Variable(inputs.cuda())
        outputs = model(inputs.unsqueeze(0))
        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        mask = target.numpy().astype(np.uint8)
        imname = dataset.masks[i].split('/')[-1]
        mask_pred = Image.fromarray(pred)
        mask_pred.putpalette(cmap)
        mask_pred.save(os.path.join('data/val', imname))
        print('eval: {0}/{1}'.format(i + 1, len(dataset)))

        inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
        inter_meter.update(inter)
        union_meter.update(union)

      iou = inter_meter.sum / (union_meter.sum + 1e-10)
      for i, val in enumerate(iou):
        print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
      print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))

  writer.close()

if __name__ == "__main__":
  main()
