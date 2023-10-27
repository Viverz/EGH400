import os
import json
import numpy as np
from PIL import Image, ImageDraw

# Set paths
JSON_DIR = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\jsonfiles'
NEW_MASKS_DIR = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\train\masks'
COLOR_MASKS_DIR = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\train\color_mask'

def json_to_mask(json_path, new_mask_path, color_mask_path):
    with open(json_path) as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    mask_image = np.zeros((image_height, image_width), dtype=np.uint8)
    color_mask_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for label_point in data['shapes']:
        points = label_point['points'][0]  # Extract the nested list of points

        x, y = int(points[0]), int(points[1])  # Access the x and y coordinates

        # Increase the size of the points
        enlarged_points = [(x-5, y-5), (x+5, y-5), (x+5, y+5), (x-5, y+5)]

        # Set the corresponding pixels in the mask to the appropriate color based on the label
        label = label_point['label']
        if label == "ground":
            fill_color = (255, 0, 0)  # Red
        elif label == "tree":
            fill_color = (0, 255, 0)  # Green
        elif label == "sky":
            fill_color = (0, 0, 255)  # Blue
        else:
            raise ValueError(f"Invalid label: {label}")

        mask_image_pil = Image.fromarray(mask_image * 255)  # Convert to binary image (0s and 255s)
        color_mask_image_pil = Image.fromarray(color_mask_image)

        draw = ImageDraw.Draw(mask_image_pil)
        draw.polygon(enlarged_points, fill=1)

        draw_color = ImageDraw.Draw(color_mask_image_pil)
        draw_color.polygon(enlarged_points, fill=fill_color)

        mask_image = np.array(mask_image_pil)
        color_mask_image = np.array(color_mask_image_pil)

    # Save the masks as binary and RGB images
    mask_image_pil.save(new_mask_path)
    color_mask_image_pil.save(color_mask_path)

# Iterate over all JSON files in the JSON directory
for json_filename in os.listdir(JSON_DIR):
    if json_filename.endswith('.json'):
        json_path = os.path.join(JSON_DIR, json_filename)
        new_mask_filename = json_filename.replace('.json', '.png')
        new_mask_path = os.path.join(NEW_MASKS_DIR, new_mask_filename)
        color_mask_filename = json_filename.replace('.json', '_color.png')
        color_mask_path = os.path.join(COLOR_MASKS_DIR, color_mask_filename)

        try:
            json_to_mask(json_path, new_mask_path, color_mask_path)
        except Exception as e:
            print(f"An error occurred while processing {json_filename}: {str(e)}")