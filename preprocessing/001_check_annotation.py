import os.path
import shutil
import numpy as np
import cv2
import argparse
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects
from config import *

import matplotlib
#matplotlib.use('Qt5Agg')  # Uses PyQt5
import matplotlib.pyplot as plt


def main(args):
    full_size_images_path = args.images_path

    if args.masks_path is None:
        full_size_masks_path = os.path.join(Path(full_size_images_path).parent.as_posix(), 'masks')
    else:
        full_size_masks_path = args.masks_path

    output_folder_contours = os.path.join(Path(full_size_masks_path).parent.as_posix(), "viz_contours")
    print('multiclass for n classes', num_classes)
    print(f'using images from {full_size_images_path} and masks from {full_size_masks_path}')

    id_classes = [i + 1 for i in range(num_classes)]
    colors = [(0, 255, 0),  # Green
              (0, 0, 255),  # Blue
              (255, 0, 0),  # Red
              (255, 255, 0),  # Yellow
              (255, 0, 255),  # Magenta
              (0, 255, 255),  # Cyan
              (128, 0, 0),  # Maroon
              (0, 128, 0)]  # Olive

    colors_dict = {1:colors[0], 2:colors[1], 3:colors[2]}

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

    fhs = os.listdir(full_size_masks_path)
    missing_images = []
    for fh in fhs:
        print(fh)

        try:
            img = cv2.imread(os.path.join(full_size_images_path, fh.replace('_mask.', '.')))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(os.path.join(full_size_masks_path, fh))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)[:, :, 0:1]
            print(f'using mask {fh}')
            #plt.imshow(gt)
            #plt.show()

            for class_value in id_classes:
                class_gt = np.uint8(gt == class_value)
                contours_mask, _ = cv2.findContours(class_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                color = colors_dict[class_value]
                cv2.drawContours(img, contours_mask, -1, color, thickness=2)
                legend = np.zeros((150, img.shape[1], 3), dtype=np.uint8)

            cv2.putText(legend, 'Mark_gt: Green', (1000, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 4)
            cv2.putText(legend, 'Graffio_gt : Blu', (1000, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 4)
            cv2.putText(legend, 'Drill_start_gt: Red', (1800, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 4)

            combined_image = np.vstack((img, legend))
            # convert to BGR again to save with cv2
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_folder_contours, fh), combined_image)
            #plt.imshow(combined_image)
            #plt.show()
        except:
            try:
                name = fh.replace('_mask.', '.')
                img = cv2.imread(os.path.join(full_size_images_path, name))
                if img is None:  # Check if the image was successfully loaded
                    raise FileNotFoundError
                print(f'black mask {fh}')
                legend = np.zeros((150, img.shape[1], 3), dtype=np.uint8)
                cv2.putText(legend, 'Mark_gt: Green', (1000, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 4)
                cv2.putText(legend, 'Graffio_gt : Blu', (1000, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255), 4)
                cv2.putText(legend, 'Drill_start_gt: Red', (1800, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 4)
                combined_image = np.vstack((img, legend))
                cv2.imwrite(os.path.join(output_folder_contours, fh), combined_image)
            except FileNotFoundError:
                print(f'No image {name}')
                missing_images.append(name)  # Store missing image name

    # Save missing images to a text file
    if missing_images:
        missing_file_path = os.path.join(Path(output_folder_contours).parent.as_posix(), "missing_images.txt")
        with open(missing_file_path, "w") as f:
            for image_name in missing_images:
                f.write(image_name + "\n")

        print(f"Missing image names saved to {missing_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--images_path", default=data_images_path, help="")
    parser.add_argument("--masks_path", default='./data/masks', help="")
    parser.add_argument("--start_from_scratch", default=1, help="")

    args = parser.parse_args()
    main(args)
