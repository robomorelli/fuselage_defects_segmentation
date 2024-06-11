import json
import shutil
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import random
import sys
sys.path.append('../')
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

    input_folder = args.images_path
    labels_input_folder = os.path.join(Path(args.images_path).parent.as_posix(), "labels")
    json_file_path = args.json_file_path

    if args.filtered or "filtered" in input_folder:
        output_folder = os.path.join(Path(args.images_path).parent.as_posix(), "filtered_masks_comparison")
    else:
        output_folder = os.path.join(Path(args.images_path).parent.as_posix(), "masks_comparison")

    # Create the output folder if it doesn't exist
    if args.start_from_scratch:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    names_to_include = ['Mark']  # classes to include into segmentation
    names_to_include = [x for x in class_names if x in names_to_include]  # revert into original order of labels
    # Create a dictionary with line numbers as keys and object names as values
    original_obj_dict = {i: name for i, name in enumerate(class_names)}
    original_class_dict = {name: i for i, name in enumerate(class_names)}
    labels_to_include = [original_class_dict[name] for name in names_to_include]

    class_dict = {key: ix for ix, key in enumerate(names_to_include)}

    num_classes = len(labels_to_include)

    # Save the dictionaries as a JSON file
    #with open(f'{json_file_path}obj_dict.json', 'w') as json_file:
    #    json.dump(obj_dict, json_file)
    with open(f'{json_file_path}class_dict.json', 'w') as json_file:
        json.dump(class_dict, json_file)

    # List all PNG files in the input folder
    png_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

    sam_checkpoint = "../models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    if not torch.cuda.is_available():
        device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Iterate through each PNG file
    for png_file in png_files:

        if (os.path.splitext(png_file)[0] + "_mask.npy") in os.listdir(output_folder):
             continue

        # load the image
        img_file = os.path.join(input_folder, png_file)

        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_res_x = image.shape[1]
        image_res_y = image.shape[0]

        predictor = SamPredictor(sam)
        predictor.set_image(image)

        txt_file = os.path.join(labels_input_folder, os.path.splitext(png_file)[0] + ".txt")
        with open(txt_file, 'r') as file:
            lines = file.readlines()

        # Initialize lists to store labels and box coordinates
        labels = []
        box_coordinates = []

        # Process each line in the file
        for line in lines:
            # Split the line into label and box coordinates
            elements = line.strip().split()
            label = int(elements[0])

            if label in labels_to_include:
                coordinates = [float(coord) for coord in elements[1:]]

                # Append label to the list
                labels.append(label)

                # Convert box coordinates to the desired format and append to the list
                x_center, y_center, width, height = coordinates
                box = [
                    int((x_center - width / 2) * image_res_x),
                    int((y_center - height / 2) * image_res_y),
                    int((x_center + width / 2) * image_res_x),
                    int((y_center + height / 2) * image_res_y)
                    ]
                box_coordinates.append(box)

        if len(labels) != 0:

            # Convert lists to torch tensors
            labels_tensor = torch.tensor(labels, dtype=torch.int64).to(device=device)
            box_coordinates_tensor = torch.tensor(box_coordinates, dtype=torch.float32).to(device=device)

            transformed_boxes = predictor.transform.apply_boxes_torch(box_coordinates_tensor, image.shape[:2])

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            image = put_bbox_on_image(image, box_coordinates, labels)

            # Map labels to object names using the dictionary
            object_names = [original_obj_dict[(label.item())] for label in labels_tensor]

            # Create a unique mask based on object names
            unique_mask = np.zeros((masks[0, 0].cpu().shape[0], masks[0, 0].cpu().shape[1], num_classes), dtype=np.uint8)

            for i, object_name in enumerate(object_names):

                mask_2d = masks[i].sum(dim=0).cpu()

                # Stack masks along the third dimension
                unique_mask[:, :, class_dict[object_name]] = unique_mask[:, :, class_dict[object_name]] + mask_2d.numpy()  # the mask of a class i is positioned at the i-th channel of the third dimension

        else:
            image = put_bbox_on_image(image, box_coordinates, labels)
            unique_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, num_classes), dtype=np.uint8)

        # Save the generated masks
        output_mask_path = os.path.join(output_folder, os.path.splitext(png_file)[0] + "_mask")
        plt.imsave(output_mask_path + '.png', np.squeeze(unique_mask), cmap='gray')
        np.save(output_mask_path, unique_mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(os.path.join(output_folder, os.path.splitext(png_file)[0]) + "_image.png"), image)

    print("Masks saved in", output_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--images_path", default=data_images_path, help="Path to the input image")
    parser.add_argument("--negative_samples_num", default=20, help="Path to the input image")
    parser.add_argument("--json_file_path", default="./json_folder", help="Path to the input image")
    parser.add_argument("--filtered", type=int, default=0, help="remove all the filtered_images into save_path dir")
    parser.add_argument("--start_from_scratch", type=int, default=0, help="remove all the filtered_images into save_path dir")

    args = parser.parse_args()
    main(args)

