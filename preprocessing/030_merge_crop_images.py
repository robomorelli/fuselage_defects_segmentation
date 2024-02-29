import os.path
import shutil
import numpy as np
import cv2
import argparse
from pathlib import Path
from config import *


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
