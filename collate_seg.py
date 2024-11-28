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


def process_folders(data_root, input_folders, output_folder):
    """Process input and target folders and combine images."""
    output_dir = os.path.join(data_root, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    input_dirs = [os.path.join(data_root, input_folder) for input_folder in input_folders]
    input_files = {}
    for id, input_dir in enumerate(input_dirs):
        input_files[input_folders[id]] = {
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if (f.endswith(".jpg") or f.endswith(".png")) and "mask" not in f
        }

    for key, value in input_files:
        # Load images
        img_paths = input_files[key]
        for img_path in img_paths:
            sam_seg_path = img_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
            robust_seg_path = img_path.replace(".jpg", "_robust_mask.png").replace(".png", "_robust_mask.png")

            if not os.path.exists(sam_seg_path) or not os.path.exists(robust_seg_path):
                print(f"Missing segmentation for {key}, skipping...")
                continue

            input_img = cv2.imread(img_path)
            input_seg = cv2.imread(sam_seg_path)
            input_robust_seg = cv2.imread(robust_seg_path)

            # Ensure all images have the same height
            images = resize_to_same_height(
                [input_img, input_seg, input_robust_seg])

            # Combine images horizontally
            combined = np.hstack(images)

            # Save combined image
            output_path = os.path.join(output_folder, f"{key}_combined.png")
            cv2.imwrite(output_path, combined)
            print(f"Saved combined image: {output_path}")


data_root = '/dataset/vfayezzhang/test/SAM/data/Test100'
input_folders = [
    'target', 'gnoise_50', 'gnoise_25', 'gnoise_15', 'input'
]
output_folder = 'output'

# Process and combine images
process_folders(data_root, input_folders, output_folder)
