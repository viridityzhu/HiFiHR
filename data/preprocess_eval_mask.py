import os
import cv2
import numpy as np

freihand_eval_root = '/home/jiayin/freihand/evaluation'
os.makedirs(os.path.join(freihand_eval_root, 'mask'), exist_ok=True)

# Get a list of all segmap files
segmap_dir = os.path.join(freihand_eval_root, 'segmap')
segmap_files = os.listdir(segmap_dir)

# Function to create a binary mask from the segmentation mask
def create_binary_mask(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Create a binary mask using the threshold value
    _, binary_mask = cv2.threshold(gray_image, 0.001, 1, cv2.THRESH_BINARY)
    return binary_mask

# Iterate through all segmap files
for segmap_file in segmap_files:
    print(segmap_file)
    # Load the segmentation mask
    segmap_path = os.path.join(segmap_dir, segmap_file)
    segmap_image = cv2.imread(segmap_path)
    
    # Create the binary mask
    binary_mask = create_binary_mask(segmap_image)
    
    # Save the binary mask in the 'mask' folder
    mask_file = os.path.join(freihand_eval_root, 'mask', segmap_file.replace('.png', '.jpg'))
    cv2.imwrite(mask_file, binary_mask * 255)

print("Binary masks generated in the 'mask' folder.")
