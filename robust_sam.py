import os
import time

import cv2
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# simple function to display the mask
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    # get the height and width from the mask
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize the pipeline for mask generation
generator = pipeline("mask-generation", model="jadechoghari/robustsam-vit-huge", device=0, points_per_batch=256)

data_root = '/dataset/vfayezzhang/test/SAM/data/Test100/input/'
file_to_handle = []

for root, dir, files in os.walk(data_root):
    for file in files:
        file_path = os.path.join(root, file)
        if (file.endswith('.jpg') or file.endswith('.png')) and 'mask' not in file_path:
            file_to_handle.append(file_path)
file_to_handle = sorted(file_to_handle)
print(file_to_handle)

for file_path in file_to_handle:
    # load the image
    image = Image.open(file_path)

    base_name = os.path.basename(file_path)
    base_prefix = os.path.splitext(base_name)[0]
    save_path = os.path.join(data_root, base_prefix + '_robust_mask.png')
    if os.path.exists(save_path):
        img = cv2.imread(save_path)
        if img is not None:
            print(f"Skip {save_path}")
            continue

    # generate the mask
    outputs = generator(image, points_per_batch=256)
    print(f"Processing {file_path}")
    # display the original image
    plt.imshow(np.array(image))
    ax = plt.gca()

    # loop through the masks and display each one
    for mask in outputs["masks"]:
        show_mask(mask, ax=ax, random_color=True)

    plt.axis("off")

    print(f"Saving to {save_path}")

    # show the image with the masks
    plt.savefig(
        save_path,
        bbox_inches='tight',  # 去掉多余的白边
        pad_inches=0  # 不留填充空间
    )
