import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

data_root = r'/dataset/vfayezzhang/test/SAM/data/Test100/target/'
file_to_handle = []
np.random.seed(0)
for root, dir, files in os.walk(data_root):
    for file in files:
        file_path = os.path.join(root, file)
        if 'mask' not in file:
            file_to_handle.append(file_path)

file_to_handle = sorted(file_to_handle)
sigmas = [15, 25, 50]
for sigma in sigmas:
    save_root = os.path.join(data_root, f'../gnoise_{sigma}')
    os.makedirs(save_root, exist_ok=True)
    for file in file_to_handle:
        print(f"Processing {file}")
        data = cv2.imread(file)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32)
        print(f"Data range: {data.min()} - {data.max()}")
        data = data / 255.0
        noise = np.random.normal(0, sigma / 255.0, data.shape)
        data = data + noise
        print(f"data shape: {data.shape}")
        masks = None

        with torch.no_grad():
            masks = mask_generator.generate(data * 255.0)

        plt.figure(figsize=(20, 20))
        plt.imshow(data)
        show_anns(masks)
        plt.axis('off')
        data = np.clip(data, 0, 1)
        data = data * 255.0
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        base_name = os.path.basename(file)
        cv2.imwrite(
            os.path.join(save_root, base_name),
            data
        )

        plt.savefig(
            os.path.join(save_root, f"{os.path.splitext(base_name)[0]}_mask.png"),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()
