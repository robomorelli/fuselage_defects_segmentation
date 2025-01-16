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



def main(args):

    input_folder = args.images_path
    output_folder = os.path.join(Path(args.images_path).parent.as_posix(), "masks")

    # Create the output folder if it doesn't exist
    if args.start_from_scratch:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # List all PNG files in the input folder
    png_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

    # Iterate through each PNG file
    for png_file in png_files:

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))

        output_mask_path = os.path.join(output_folder, os.path.splitext(png_file)[0] + "_mask")
        plt.imsave(output_mask_path + '.png', np.squeeze(mask), cmap='gray')

    print("Masks saved in", output_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--images_path", default=may_data_images_path, help="Path to the input image")
    parser.add_argument("--start_from_scratch", default=1, help="Path to the input image")

    args = parser.parse_args()
    main(args)
