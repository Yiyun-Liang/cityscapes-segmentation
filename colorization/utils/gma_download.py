import cv2
import numpy as np
import os
import urllib.request
import json
import pandas as pd

from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='Download Google Maps Data')
parser.add_argument('--csv_dir', default='data/', help='directory that contains the coordinates of cityscapes samples')
parser.add_argument('--out_dir', default='data/', help='directory to store the google maps images')
parser.add_argument('--api_key', help='directory to store the google maps images')
parser.add_argument('--zoom_level', type=int, default='18', help='directory to store the google maps images')
parser.add_argument('--width', type=int, default=224, help='final width of the images')
parser.add_argument('--height', type=int, default=224, help='final height of the images')
args = parser.parse_args()

def sample_image(out_fn, lat, lon, api_key, height=520, width=500, zoom=18):
    """
    This function uses the Google Static Maps API to download and save
    one satellite image.
    :param out_fn: Output filename for saved image
    :param lat: Latitude of image center
    :param lon: Longitude of image center
    :param height: Height of image in pixels
    :param width: Width of image in pixels
    :param zoom: Zoom level of image
    :return: True if valid image saved, False if no image saved
    """
    # Google Static Maps API key
    try:
        # Save extra tall satellite image
        height_buffer = 100
        url_pattern = 'https://maps.googleapis.com/maps/api/staticmap?center=%0.6f,%0.6f&zoom=%s&size=%sx%s&maptype=satellite&key=%s'
        url = url_pattern % (lat, lon, zoom, width, height + height_buffer, api_key)
        urllib.request.urlretrieve(url, out_fn)

        # Cut out text at the bottom of the image
        image = cv2.imread(out_fn)
        image = image[int((height_buffer/2)):int((height+height_buffer/2)),:,:]
        image = cv2.resize(image, (width, height))
        cv2.imwrite(out_fn, image)

        # Check file size and delete invalid images < 10kb
        fs = os.stat(out_fn).st_size
        if fs < 10000:
            os.remove(out_fn)
            return False
        else:
            return True

    except:
        return False

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

df = pd.read_csv(args.csv_dir)
for ind, row in df.iterrows():
    output_location = '{}{}.jpg'.format(args.out_dir, row['location'].split('/')[-1].split('.json')[0])
    sample_image(output_location, row['lat'], row['lon'], args.api_key, args.height, args.width, args.zoom_level)
