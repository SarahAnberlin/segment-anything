import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    for img_id in sorted(next(iter(input_files.values()))):  # Iterate over IDs from any folder
        print(f"Processing ID: {img_id}")
        rows = []
        fig, axes = plt.subplots(len(input_folders), 1, figsize=(15, 5 * len(input_folders)))
        fig.tight_layout(pad=5)

        for key, img_paths in input_files.items():
            img_path = next((path for path in img_paths if img_id in path), None)
            if not img_path:
                print(f"Missing image for {key} and ID {img_id}, skipping...")
                continue

            sam_seg_path = img_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
            robust_seg_path = img_path.replace(".jpg", "_robust_mask.png").replace(".png", "_robust_mask.png")

            if not os.path.exists(sam_seg_path) or not os.path.exists(robust_seg_path):
                print(f"Missing segmentation for {key}, skipping...")
                continue

            input_img = cv2.imread(img_path)
            input_seg = cv2.imread(sam_seg_path)
            input_robust_seg = cv2.imread(robust_seg_path)

            # Ensure all images have the same height
            images = resize_to_same_height([input_img, input_seg, input_robust_seg])

            # Combine images horizontally
            row = np.hstack(images)
            rows.append(row)

            # Plotting the combined row
            ax = axes[input_folders.index(key)]
            ax.imshow(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{key}")
            ax.axis("off")

        # Save the combined figure
        output_path = os.path.join(output_dir, f"{img_id}_combined.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved combined figure: {output_path}")


data_root = '/dataset/vfayezzhang/test/SAM/data/Test100'
input_folders = [
    'target', 'gnoise_50', 'gnoise_25', 'gnoise_15', 'input'
]
output_folder = 'output'

# Process and combine images
process_folders(data_root, input_folders, output_folder)
