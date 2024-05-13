import argparse
import os
import shutil
from pathlib import Path
import shutil
from config import *


def main(args):

    dest_path = args.renamed_masks_path
    source_path = os.path.join(Path(dest_path).parent.as_posix(), 'masks')

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
        os.makedirs(dest_path)
    else:
        os.makedirs(dest_path)

    filenames = os.listdir(source_path)

    for fh in filenames:
        shutil.copyfile(os.path.join(source_path, fh), os.path.join(dest_path, fh.replace('_mask.', '.')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--renamed_masks_path", default=april_data_renamed_masks_path , help="Path to the input image")

    args = parser.parse_args()
    main(args)

