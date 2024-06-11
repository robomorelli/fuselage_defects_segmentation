import json
import os
import shutil
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import yaml
import sys
sys.path.append('../')
from config import *

def main(args):
    remapping_dict = args.remapping_dict
    multiclass_masks_folder = args.multiclass_masks_path
    folds = os.listdir(multiclass_masks_folder)

    input_masks_path = args.input_masks_path
    input_images_path = args.input_images_path

    out_masks_path = os.path.join(Path(input_masks_path).parent.as_posix(), "masks_remapped")
    out_viz_masks_path = os.path.join(Path(input_masks_path).parent.as_posix(), "masks_remapped_viz")
    npy_files = os.listdir(input_masks_path)
    images_files = os.listdir(input_images_path)
    images_files = [x for x in images_files if '.png' in x]
    l1 = len(images_files)
    images_files = [x for x in images_files if x.replace('png', 'npy') in npy_files]
    l2 = len(images_files)

    print(f'starting images {l1}, images with masks {l2}')
    if not os.path.exists(out_masks_path):
        os.makedirs(out_masks_path)

    if not os.path.exists(out_viz_masks_path):
        os.makedirs(out_viz_masks_path)

    with open('class_mapping_viz.yaml', "r") as f:
        mapping_dict_viz = yaml.safe_load(f)

    # Iterate through each PNG file
    for npy_file in npy_files:

        # load the image
        npy_img_file = os.path.join(input_masks_path, npy_file)
        npy_image = np.load(npy_img_file)

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
        mask_viz = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
        for ix, class_name in enumerate(folds):
            mask[npy_image == ix] = remapping_dict[str(ix)]
            mask_viz[npy_image == ix] = mapping_dict_viz[class_name]

            print(os.path.join(out_viz_masks_path, npy_file.replace('.npy', '.png')))

            cv2.imwrite(os.path.join(out_viz_masks_path, npy_file.replace('.npy', '_mask.png')), np.squeeze(mask_viz))
            cv2.imwrite(os.path.join(out_masks_path, npy_file.replace('.npy', '_mask.png')), np.squeeze(mask))

            if args.move_to_main_folders:
                cv2.imwrite(os.path.join(masks_viz_path, npy_file.replace('.npy', '_mask.png'))
                            , np.squeeze(mask_viz))
                cv2.imwrite(os.path.join(data_masks_path,
                                         npy_file.replace('.npy', '_mask.png')), np.squeeze(mask))
                shutil.copy(os.path.join(input_images_path, npy_file.replace('.npy', '.png')),
                            data_images_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--input_masks_path", default=masks_to_remap_path, help="Path to the input image")
    parser.add_argument("--input_images_path", default=images_2_wave_path, help="Path to the input image")
    parser.add_argument("--multiclass_masks_path", default=multiclass_masks_path, help="Path to the input image")
    parser.add_argument("--move_to_main_folders", default=1, help="")

    parser.add_argument("--remapping_dict", default={'0':1, '1':2, '2':0}, help="Path to the input image")
    args = parser.parse_args()
    main(args)

