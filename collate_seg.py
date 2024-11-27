import os
import cv2
import numpy as np


def resize_to_same_height(images):
    """Resize all images to the same height while maintaining aspect ratio."""
    target_height = min(img.shape[0] for img in images)
    resized_images = [
        cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
        for img in images
    ]
    return resized_images


def process_folders(input_folder, target_folder, output_folder):
    """Process input and target folders and combine images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ensure files are paired based on names
    input_files = {os.path.splitext(f)[0]: os.path.join(input_folder, f)
                   for f in os.listdir(input_folder)
                   if f.endswith(('.png', '.jpg')) and 'mask' not in f}
    target_files = {os.path.splitext(f)[0]: os.path.join(target_folder, f)
                    for f in os.listdir(target_folder)
                    if f.endswith(('.png', '.jpg')) and 'mask' not in f}

    common_keys = input_files.keys() & target_files.keys()

    for key in common_keys:
        # Load images
        input_img_path = input_files[key]
        target_img_path = target_files[key]
        input_seg_path = input_img_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
        target_seg_path = target_img_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
        input_robust_seg_path = input_img_path.replace(".jpg", "_robust_mask.png").replace(".png", "_robust_mask.png")
        target_robust_seg_path = target_img_path.replace(".jpg", "_robust_mask.png").replace(".png", "_robust_mask.png")

        if not os.path.exists(input_seg_path) or not os.path.exists(target_seg_path):
            print(f"Missing segmentation for {key}, skipping...")
            continue

        input_img = cv2.imread(input_img_path)
        target_img = cv2.imread(target_img_path)
        input_seg = cv2.imread(input_seg_path)
        target_seg = cv2.imread(target_seg_path)
        input_robust_seg = cv2.imread(input_robust_seg_path)
        target_robust_seg = cv2.imread(target_robust_seg_path)

        # Ensure all images have the same height
        images = resize_to_same_height(
            [target_img, target_seg, target_robust_seg, input_img, input_seg, input_robust_seg])

        # Combine images horizontally
        combined = np.hstack(images)

        # Save combined image
        output_path = os.path.join(output_folder, f"{key}_combined.png")
        cv2.imwrite(output_path, combined)
        print(f"Saved combined image: {output_path}")


# Define folder paths
input_folder = r'C:\Users\smartmore\Desktop\Test100\input'
target_folder = r'C:\Users\smartmore\Desktop\Test100\target'
output_folder = r'C:\Users\smartmore\Desktop\Test100\output'

# Process and combine images
process_folders(input_folder, target_folder, output_folder)
