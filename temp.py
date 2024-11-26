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

data_root = r'C:\Users\smartmore\Desktop\Test100'
for root, dir, files in os.walk(data_root):
    for file in files:
        file_path = os.path.join(root, file)
        if 'mask' in file:
            # Delete
            os.remove(file_path)

for root, dir, files in os.walk(data_root):
    for file in files:
        file_path = os.path.join(root, file)
        if (file.endswith('.png') or file.endswith('.jpg')):
            data_path = os.path.join(root, file)
            print(f"Processing {data_path}")
            data = cv2.imread(data_path)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            masks = None

            with torch.no_grad():
                masks = mask_generator.generate(data)

            plt.figure(figsize=(20, 20))
            plt.imshow(data)
            show_anns(masks)
            plt.axis('off')
            plt.savefig(
                os.path.join(root, f"{file}_mask.png"),
                bbox_inches='tight',  # 去掉多余的白边
                pad_inches=0  # 不留填充空间
            )
