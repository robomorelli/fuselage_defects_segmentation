import os.path
import shutil
import numpy as np
import cv2
import argparse
from pathlib import Path
from PIL import Image
from skimage.morphology import remove_small_holes, remove_small_objects
from config import *

# Function to keep one of the masks for tha same image


# Function to merge 2 images and keep only one of them with only one name
def merge_images_masks(merge_images_txt_folder_path, full_size_images_path, full_size_masks_path, keep_first=True):

    with open(merge_images_txt_folder_path, 'r') as file:
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
    if args.merge_images_folder is not None:
        merge_images_masks(args.merge_images_folder, full_size_images_path, full_size_masks_path, keep_first=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--full_size_images_path", default=data_images_path, help="")
    parser.add_argument("--generate_black_masks", default=1, help="")
    parser.add_argument("--merge_images_folder", default='./images_to_merge.txt', help="")
    parser.add_argument("--add_new_class", default='./data/raw_masks_drill_start', help="")

    args = parser.parse_args()
    main(args)
