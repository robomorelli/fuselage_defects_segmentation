import os.path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from config import *

# Function to keep one of the masks for tha same image


# Function to merge 2 images and keep only one of them with only one name
def merge_images_masks(merge_images_txt_path, full_size_images_path, full_size_masks_path, keep_first=True):

    with open(merge_images_txt_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

        for l in lines:
            m1_name, m2_name = l.split(';')[0], l.split(';')[1]
            i1_name = m1_name if 'png' in m1_name else m1_name + '.png'
            i2_name = m2_name if 'png' in m2_name else m2_name + '.png'
            ifh1 = os.path.join(full_size_images_path, i1_name)
            ifh2 = os.path.join(full_size_images_path, i2_name)
            mfh1 = os.path.join(full_size_masks_path, m1_name.replace('.', '_mask.'))
            mfh2 = os.path.join(full_size_masks_path, m2_name.replace('.', '_mask.'))
            m1 = cv2.imread(mfh1)
            m1 = cv2.cvtColor(m1, cv2.COLOR_BGR2RGB)[:, :, 0:1]
            m2 = cv2.imread(mfh2)
            m2 = cv2.cvtColor(m2, cv2.COLOR_BGR2RGB)[:, :, 0:1]

            m = np.maximum(m1, m2)

            if keep_first:
                os.remove(mfh2)
                os.remove(ifh2)
                cv2.imwrite(mfh1, m)
            else:
                os.remove(mfh1)
                os.remove(ifh1)
                cv2.imwrite(mfh2, m)


def add_new_class(new_class_masks_path, original_masks_path, test_path=None, force_recreate=True):
    """
    Adds new class masks from `new_class_masks_path` to the masks in `full_size_masks_path` if the class ID doesn't already exist.

    Args:
        new_class_masks_path (str): Path to the new class masks.
        original_masks_path (str): Path to the original masks where the new class will be added.
        test_path (str, optional): Path to save the updated masks for testing purposes. If None, it doesn't save. Defaults to None.
    """
    if test_path is not None:
        os.makedirs(test_path, exist_ok=True)
        output_path = test_path
    else:
        output_path = original_masks_path

    # Get the list of mask filenames in both directories
    new_class_mask_files = os.listdir(new_class_masks_path)
    missing_masks = []
    # Iterate through the new class masks
    for new_mask_file in tqdm(new_class_mask_files):
        print(new_mask_file)
        new_mask_path = os.path.join(new_class_masks_path, new_mask_file)
        new_mask = np.squeeze(cv2.imread(new_mask_path, cv2.IMREAD_UNCHANGED)[:,:,0:1])
        original_mask_path = os.path.join(original_masks_path, new_mask_file)
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_UNCHANGED)

        try:
            if original_mask is None:  # Check if the image was successfully loaded
                print(f'Original mask not existing: {new_mask_file}')
                raise FileNotFoundError
        except:
            original_mask = np.zeros_like(new_mask)
            missing_masks.append(new_mask_file)

        if len(original_mask.shape)>2:
            original_mask = np.squeeze(original_mask[:,:,0:1])

        #new_mask[new_mask == 255] = 0
        #original_mask[original_mask == 255] = 0
        unique_values1 = np.unique(new_mask)
        unique_values2 = np.unique(original_mask)
        unique_values1 = unique_values1[unique_values1 != 0]
        unique_values2 = unique_values2[unique_values2 != 0]
        if np.intersect1d(unique_values1, unique_values2):
            print(f'new class already present in {new_mask_file}')
            if force_recreate:
                for id in unique_values2:
                    original_mask[original_mask==id] = 0
            else:
                continue

        # Ensure the new mask has the same size as the original
        if original_mask.shape != new_mask.shape:
            raise ValueError(
                f"Mask shape mismatch: {original_mask.shape} and {new_mask.shape} do not match for {new_mask_file}.")

        # Add the new mask to the original mask
        original_mask = np.maximum(original_mask, new_mask)

        # If test_path is provided, save a copy of the updated mask to the test path
        cv2.imwrite(os.path.join(output_path, new_mask_file), original_mask)
        print(f"Saved updated mask to test folder: {output_path}")

    # Save missing images to a text file
    if missing_masks:
        missing_file_path = os.path.join(Path(output_path).parent.as_posix(), "missing_masks.txt")
        with open(missing_file_path, "w") as f:
            for image_name in missing_masks:
                f.write(image_name + "\n")

        print(f"Missing image names saved to {missing_file_path}")
    print("Process completed.")



# Function to create black mask for not annotated images (Actually done  during the crop and only for the crop not for full size images)
def generate_black_masks(full_size_images_path, full_size_masks_path):

    iname = os.listdir(full_size_images_path)
    mname = os.listdir(full_size_masks_path)

    missing_images = [x for x in mname if x.replace('_mask.', '.') not in iname]

    for mi in missing_images:
        os.remove(os.path.join(full_size_masks_path, mi))

    missing_masks = [x.replace('.', '_mask.') for x in iname if x.replace('.', '_mask.') not in mname]

    for mm in missing_masks:
        msk = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0))
        msk.save(os.path.join(full_size_masks_path, mm))


def main(args):
    full_size_images_path = args.full_size_images_path
    full_size_masks_path = os.path.join(Path(args.full_size_images_path).parent.as_posix(), 'masks')

    if args.generate_black_masks:
        generate_black_masks(full_size_images_path, full_size_masks_path)
    if args.merge_images_path is not None:
        merge_images_masks(args.merge_images_path, full_size_images_path, full_size_masks_path, keep_first=True)
    if args.new_class_path is not None:
        add_new_class(args.new_class_path, args.original_class_path, args.test_path, force_recreate=args.force_recreate)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--full_size_images_path", default=data_images_path, help="")
    parser.add_argument("--generate_black_masks", default=0, help="")
    parser.add_argument("--merge_images_path", default=None, help="./images_to_merge.txt - is the folder including txt of images name to merge")
    parser.add_argument("--original_class_path", default=data_masks_path, help="")
    parser.add_argument("--new_class_path", default='../data/raw_masks_drill_start', help="")
    parser.add_argument("--test_path", default=None, help="")
    parser.add_argument("--force_recreate", default=1, help="")

    args = parser.parse_args()
    main(args)
