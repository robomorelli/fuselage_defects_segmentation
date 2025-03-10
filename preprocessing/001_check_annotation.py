import os.path
import shutil
import numpy as np
import cv2
import argparse
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects
from config import *


def main(args):
    full_size_images_path = args.full_size_images_path
    full_size_masks_path = os.path.join(Path(args.full_size_images_path).parent.as_posix(), 'masks')
    #full_size_images_path = "../data/images_2_wave"
    #full_size_masks_path = "../data/masks_remapped"
    output_folder_contours = os.path.join(Path(full_size_masks_path).parent.as_posix(), "viz_contours")

    print('multiclass for n classes', num_classes)

    id_classes = [i + 1 for i in range(num_classes)]
    colors = [(0, 255, 0),  # Green
              (0, 0, 255),  # Red
              (255, 0, 0),  # Blue
              (255, 255, 0),  # Cyan
              (255, 0, 255),  # Magenta
              (0, 255, 255),  # Yellow
              (128, 0, 0),  # Maroon
              (0, 128, 0)]  # Olive

    # Create the output folder if it doesn't exist
    if args.start_from_scratch:
        if os.path.exists(output_folder_contours):
            shutil.rmtree(output_folder_contours)
            os.makedirs(output_folder_contours, exist_ok=True)
        else:
            os.makedirs(output_folder_contours, exist_ok=True)
    else:
        os.makedirs(output_folder_contours, exist_ok=True)
    id_classes = [i + 1 for i in range(num_classes)]

    fhs = os.listdir(full_size_images_path)
    for fh in fhs:

        try:
            img = cv2.imread(os.path.join(full_size_images_path, fh))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(os.path.join(full_size_masks_path, fh.replace('.', '_mask.')))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)[:, :, 0:1]

            for class_value in id_classes:
                class_gt = np.uint8(gt == class_value)
                contours_mask, _ = cv2.findContours(class_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                color = (0, 128, 0) if class_value == 1 else (255, 255, 0)
                cv2.drawContours(img, contours_mask, -1, color, thickness=2)
                legend = np.zeros((150, img.shape[1], 3), dtype=np.uint8)

            cv2.putText(legend, 'Mark_gt: Yellow', (1000, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 128, 0), 4)
            cv2.putText(legend, 'Graffio_gt : Cyan', (1000, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 255, 0), 4)

            combined_image = np.vstack((img, legend))
            # convert to BGR again to save with cv2
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
            #cropped_0000_label_0xjUoY7D_1440_480
            cv2.imwrite(os.path.join(output_folder_contours, fh), combined_image)
        except:
            img = cv2.imread(os.path.join(full_size_images_path, fh))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f'black mask {fh}')
            cv2.imwrite(os.path.join(output_folder_contours, fh), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--full_size_images_path", default=cropped_images_path, help="")
    parser.add_argument("--start_from_scratch", default=1, help="")

    args = parser.parse_args()
    main(args)
