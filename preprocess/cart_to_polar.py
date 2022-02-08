'''
This script is used for KidnappedRadar: convert Cartesian birdview images to Polar birdview images
'''
# %%
import cv2
import math
import tqdm
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='../dataset/7n5s_xy11/img')
args = parser.parse_args()


def cart_to_polar(input_dir, output_dir):
    img = cv2.imread(input_dir, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    maxRadius = math.hypot(w / 2, h / 2)
    linear_polar = cv2.linearPolar(img, (w / 2, h / 2), maxRadius, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
    cv2.imwrite(output_dir, linear_polar)
    return 0


# %%
STRUCT_DIR = args.dataset
CART_DIR = os.path.join(STRUCT_DIR, 'img')
POLAR_DIR = os.path.join(STRUCT_DIR, 'img_polar')
if not os.path.exists(POLAR_DIR):
    os.mkdir(POLAR_DIR)

imgs = os.listdir(CART_DIR)
for img in tqdm.tqdm(imgs):
    input_dir = os.path.join(CART_DIR, img)
    output_dir = os.path.join(POLAR_DIR, img)
    cart_to_polar(input_dir, output_dir)
