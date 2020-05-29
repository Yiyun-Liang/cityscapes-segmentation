import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import cv2

class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)


def preprocess(image, mask, flip=False, scale=None, crop=None):
  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
  if scale:
    w, h = image.size
    rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    image = image.resize(new_size, Image.ANTIALIAS)
    mask = mask.resize(new_size, Image.NEAREST)

  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  image = data_transforms(image)
  mask = torch.LongTensor(np.array(mask).astype(np.int64))

  if crop:
    h, w = image.shape[1], image.shape[2]
    pad_tb = max(0, crop[0] - h)
    pad_lr = max(0, crop[1] - w)
    image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
    mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

    h, w = image.shape[1], image.shape[2]
    i = random.randint(0, h - crop[0])
    j = random.randint(0, w - crop[1])
    image = image[:, i:i + crop[0], j:j + crop[1]]
    mask = mask[i:i + crop[0], j:j + crop[1]]
  return image, mask

def get_ratio(seg_map, target=False, ignore_class=255):
  class_names = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
  ]
  num_classes = len(class_names)
  train_id_map = dict(zip(np.arange(0,num_classes), class_names))
  
  batch_size = seg_map.shape[0]
  if not target:
    arr = F.softmax(seg_map, dim=1)
    arr = torch.sum(arr, dim=(2,3))/(seg_map.shape[2]*seg_map.shape[3])
  else:
    arr = torch.zeros((batch_size, num_classes))
    for cl in range(num_classes):
      for bs in range(batch_size):
        arr[bs, cl] = torch.true_divide(torch.sum(seg_map[bs, :, :] == cl), (seg_map.shape[1]*seg_map.shape[2]))
        #print(cl, torch.sum(seg_map[bs, :, :] == cl))
  return arr

def get_moments(image):
  # calculate moments of binary image
  image = image.astype(np.uint8)
  M = cv2.moments(image)

  if M["m00"] != 0:
    # calculate x,y coordinate of center
    cX = float(int(M["m10"] / M["m00"]))
    cY = float(int(M["m01"] / M["m00"]))
  else:
    cX = float(0)
    cY = float(0)

  return cX, cY

# reference: https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
def get_centroid(seg_map, target=False, ignore_class=255):
  class_names = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
  ]
  num_classes = len(class_names)
  train_id_map = dict(zip(np.arange(0,num_classes), class_names))

  batch_size = seg_map.shape[0]
  image = seg_map
  if not target:
    image = torch.argmax(seg_map, axis=1)
  image = image.cpu().detach().numpy()
  # convert the grayscale image to binary image
  # ret,thresh = cv2.threshold(image,127,255,0)

  res = []
  for n in range(batch_size):
    moments = []
    single = image[n]
    for i in range(0, num_classes):
      cur = single.copy()
      if i==0:
        cur[cur==i] = 100
        cur[cur!=100] = 0
      else:
        cur[cur!=i] = 0
      cX, cY = get_moments(np.array(cur))
      moments.append([cX, cY])
    res.append(moments)

  res = np.array(res)
  return torch.from_numpy(res)

def get_adj_matrix(seg_map, target=False, ignore_class=255):
  pass
