import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Set the path to your dataset
dataset_path = '../datasets/USIS10K/test/'
ann_file = os.path.join('../datasets/USIS10K/', 'multi_class_annotations', 'multi_class_test_annotations.json')  # Adjust filename as needed
output_dir = os.path.join(dataset_path, 'GT')  # Folder to save annotated images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(ann_file)

# Define your classes and custom color palette
CLASSES = (
    'wrecks/ruins', 'fish', 'reefs',
    'aquatic plants', 'human divers',
    'robots', 'sea-floor'
)
PALETTE = [
    (220, 20, 60),  # wrecks/ruins
    (255, 0, 0),    # fish
    (0, 0, 142),    # reefs
    (0, 0, 70),     # aquatic plants
    (0, 60, 100),   # human divers
    (0, 80, 100),   # robots
    (0, 0, 230)     # sea-floor
]

# Map class names to IDs
color_palette = {i + 1: PALETTE[i] for i in range(len(CLASSES))}  # COCO uses 1-based indexing for category IDs

# Get image ids and loop through them
image_ids = coco.getImgIds()
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(dataset_path, img_info['file_name'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a mask image for segmentation
    mask_image = np.zeros(image.shape, dtype=np.uint8)

    # Load annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Draw bounding boxes and masks
    for ann in annotations:
        class_id = ann['category_id']
        if 'bbox' in ann:
            x, y, width, height = ann['bbox']
            cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)

        if 'segmentation' in ann:
            for seg in ann['segmentation']:
                seg = np.array(seg).reshape(-1, 2).astype(np.int32)
                color = color_palette.get(class_id, (0, 0, 0))  # Default to black if class_id not found
                cv2.fillPoly(mask_image, [seg], color)

    # Combine original image and mask
    combined_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

    # Save the annotated image
    output_path = os.path.join(output_dir, f'gt_{img_info["file_name"]}')
    cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

