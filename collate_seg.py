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
    input_files = {folder: [] for folder in input_folders}

    # Gather all image paths for each folder
    for id, input_dir in enumerate(input_dirs):
        input_files[input_folders[id]] = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if (f.endswith(".jpg") or f.endswith(".png")) and "mask" not in f
        ]

    # Process each unique image ID based on the target folder
    for img_path in input_files['target']:
        img_id = os.path.basename(img_path)  # Extract the unique ID from the file name
        rows = []
        fig, axes = plt.subplots(len(input_folders), 1, figsize=(15, 5 * len(input_folders)))
        fig.tight_layout(pad=5)

        for idx, key in enumerate(input_folders):
            # Construct paths for the corresponding images in each folder
            img_path = os.path.join(data_root, key, img_id)
            sam_seg_path = img_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
            robust_seg_path = img_path.replace(".jpg", "_robust_mask.png").replace(".png", "_robust_mask.png")

            if not os.path.exists(img_path):
                print(f"Missing image for {key}: {img_id}, skipping...")
                continue
            if not os.path.exists(sam_seg_path) or not os.path.exists(robust_seg_path):
                print(f"Missing segmentation for {key}, skipping...")
                continue

            # Read the images
            input_img = cv2.imread(img_path)
            input_seg = cv2.imread(sam_seg_path)
            input_robust_seg = cv2.imread(robust_seg_path)

            # Resize all images to the same height
            images = resize_to_same_height([input_img, input_seg, input_robust_seg])

            # Combine images horizontally
            row = np.hstack(images)
            rows.append(row)

            # Plot the combined row
            ax = axes[idx]
            ax.imshow(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{key}")
            ax.axis("off")

        # Save the combined figure
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_id)[0]}_combined.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved combined figure: {output_path}")


# Parameters
data_root = '/dataset/vfayezzhang/test/SAM/data/Test100'
input_folders = [
    'target', 'gnoise_50', 'gnoise_25', 'gnoise_15', 'input'
]
output_folder = 'output'

# Process and combine images
process_folders(data_root, input_folders, output_folder)
