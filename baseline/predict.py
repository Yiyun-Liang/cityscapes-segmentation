import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

test_dir_img = 'data/imgs/test/'
test_dir_mask = 'data/masks/test/'
test_output_imgs = 'output/'

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):

    test_dataset = BasicDataset(test_dir_img, test_dir_mask, split='test')
    n_test = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    # net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long

    test_net(net, test_loader, device)

    # with tqdm(total=n_test, desc='Test round', unit='batch', leave=False) as pbar:
    #     for batch in loader:
    #         imgs, true_masks = batch['image'], batch['mask']
    #         imgs = imgs.to(device=device, dtype=torch.float32)
    #         true_masks = true_masks.to(device=device, dtype=mask_type)

    #         with torch.no_grad():
    #             mask_pred = net(imgs)

    #         probs = F.softmax(mask_pred, dim=1)
    #         probs = probs.squeeze(0)
    #         full_mask = probs.squeeze().cpu().numpy()

    #         pbar.update()

def decode_segmap(img, label_colours):
    img = img.transpose(1,2,0)
    img = np.argmax(img, axis=2)
    r = img.copy()
    g = img.copy()
    b = img.copy()
    for l in range(0, 19):
        r[img == l] = label_colours[l][0]
        g[img == l] = label_colours[l][1]
        b[img == l] = label_colours[l][2]
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

colors = [ [128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],[0, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32],]

label_colours = dict(zip(range(19), colors))

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=19)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = cv2.imread(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(decode_segmap(mask, label_colours))
        #     result.save(out_files[i])

        #     logging.info("Mask saved to {}".format(out_files[i]))

        # if args.viz:
        #     logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        #     plot_img_and_mask(img, mask)
