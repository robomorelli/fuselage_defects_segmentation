import json
import os.path
import shutil
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import random
from config import *


def generate_coordinates_outside_bbox(coordinates, num_points):
    xmin, ymin, xmax, ymax = coordinates
    outside_coordinates = []

    for _ in range(num_points):
        x = random.randint(0, xmin) if random.choice([True, False]) else random.randint(xmax, IMG_WIDTH)
        y = random.randint(0, ymin) if random.choice([True, False]) else random.randint(ymax, IMG_HEIGHT)

        outside_coordinates.append((x, y))

    return outside_coordinates


def put_bbox_on_image(image, coordinates, labels):

    for box, label in zip(coordinates, labels):
        x1, y1, x2, y2 = box
        left = x1
        top = y1
        right = x2
        bottom = y2

        # Draw bounding box
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

        # Draw label
        label_text = str(label)
        if class_names is not None:
            label_text = class_names[label]

        label_position = (left, top - 10)  # Adjust label position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(image, label_text, label_position, font, font_scale, color, font_thickness)

    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def main(args):

    input_folder = args.crops_path
    suffix = os.path.basename(input_folder)
    output_folder = os.path.join(Path(args.crops_path).parent.as_posix(), f"merged_{suffix}")

    # Create the output folder if it doesn't exist
    if args.start_from_scratch:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    crop_filenames = os.listdir(input_folder)
    if "mask" in crop_filenames[0]:
        crop_filenames = [x.replace('_mask.', '.') for x in crop_filenames]
        mask = True
    else:
        mask = False
    crop_filenames = [x.replace('.png', '') for x in crop_filenames]

    for cfh in crop_filenames:
        basename = '_'.join(cfh.split('_')[:-2])
        crop_to_merge = [crop_fh for crop_fh in crop_filenames if '_'.join(crop_fh.split('_')[:-2]) == basename]

        x_splits = [int(crop_fh.split('_')[-2]) for crop_fh in crop_to_merge]
        y_splits = [int(crop_fh.split('_')[-1]) for crop_fh in crop_to_merge]

        x_splits = np.unique(x_splits)
        y_splits = np.unique(y_splits)

        x_splits.sort()
        y_splits.sort()

        image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))

        for ixs, x_s in enumerate(x_splits[:-1]):
            for iys, y_s in enumerate(y_splits[:-1]):
                if mask:
                    crop_path = os.path.join(input_folder, (basename + f"_{x_s}" + f"_{y_s}_mask.png"))
                else:
                    crop_path = os.path.join(input_folder, (basename + f"_{x_s}" + f"_{y_s}.png"))
                crop = cv2.imread(crop_path)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                #crop = crop / 255.0
                #image[x_s: x_splits[ixs+1], y_s : y_splits[iys+1]] = crop
                image[y_s: y_splits[iys + 1], x_s: x_splits[ixs + 1]] = crop

        cv2.imwrite(os.path.join(output_folder, f"{basename.lstrip('cropped_')}.png"), np.squeeze(image))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--crops_path", default=os.path.join(common_path_test_results, f"model_results_{0.45}"), help="")
    parser.add_argument("--start_from_scratch", default=1, help="")

    args = parser.parse_args()
    main(args)
