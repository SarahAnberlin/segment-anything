from transformers import pipeline
from PIL import Image

# initialize the pipeline for mask generation
generator = pipeline("mask-generation", model="jadechoghari/robustsam-vit-huge", device=0, points_per_batch=256)

image_url = r'C:\Users\smartmore\Desktop\Test100\input\1.jpg'
outputs = generator(image_url, points_per_batch=256)

import matplotlib.pyplot as plt
from PIL import Image
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


# display the original image
plt.imshow(np.array(Image.open(image_url)))
ax = plt.gca()

# loop through the masks and display each one
for mask in outputs["masks"]:
    show_mask(mask, ax=ax, random_color=True)

plt.axis("off")

# show the image with the masks
plt.show()
