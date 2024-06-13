import os.path
import shutil
import argparse
from pathlib import Path
from config import *


def main(args):

    input_masks_path = args.input_masks_path
    input_images_path = os.path.join(Path(input_masks_path).parent.as_posix(), "images")
    removed_masks_path = os.path.join(Path(input_masks_path).parent.as_posix(), 'removed_masks')
    removed_images_path = os.path.join(Path(input_masks_path).parent.as_posix(), "removed_images")

    if not os.path.exists(removed_masks_path):
        os.makedirs(removed_masks_path)
    if not os.path.exists(removed_images_path):
        os.makedirs(removed_images_path)

    with open(args.wrong_masks_filepath, 'r') as fh:
        for line in fh:
            mask_name = line.strip()
            image_name = mask_name.replace('_mask.', '.')
            try:
                shutil.move(os.path.join(input_masks_path, mask_name), os.path.join(removed_masks_path, mask_name))
                shutil.move(os.path.join(input_images_path, image_name), os.path.join(removed_images_path, image_name))
            except:
                print(f'problem with {image_name} mask/image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--input_masks_path", default=data_masks_path, help="")
    parser.add_argument("--wrong_masks_filepath", default='../data/wrong_masks.txt', help="")

    args = parser.parse_args()
    main(args)
