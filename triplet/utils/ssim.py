import os
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from SSIM_PIL import compare_ssim
import glob
import sys
test_dir = '/atlas/u/buzkent/MMVideoPredictor/cv/saved_images_all/'

files = glob.glob(test_dir + '*_real.jpg')

ssim_score = 0
for file in files:
    target_path = file
    pred_path = file[:-8] + 'pred.jpg'
    print(pred_path)
    
    target = Image.open(target_path).convert('L')
    pred = Image.open(pred_path).convert('L')
    
#     ssim_score += ssim(pred, target, data_range=255, size_average=False)
    ssim_score += compare_ssim(pred, target)
    print(ssim_score)
    
print(ssim_score/len(files))