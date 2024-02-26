import matplotlib
import numpy as np
import sys
sys.path.append('..')
from config import *
matplotlib.use('Qt5Agg')
from PIL import Image
import argparse
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from config import *

def has_positive_pixel(mask_path):
    with Image.open(mask_path) as mask:
        # Convert the image to grayscale
        mask = mask.convert('L')

        # Get pixel values as a list
        pixel_values = list(mask.getdata())

        # Check if there is at least one pixel greater than zero
        return any(value > 0 for value in pixel_values)

def crop_images(args):

    # Example usage
    images_input_folder = args.images_path
    masks_input_folder = os.path.join(Path(images_input_folder).parent.as_posix(), 'masks')
    save_bkg_perc = args.save_bkg_perc
    if args.total_background:
        image_output_folder = os.path.join(Path(images_input_folder).parent.as_posix(), 'cropped_data/tot_bkg/images')
        mask_output_folder = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data/tot_bkg/masks')
        save_bkg_perc = 1.00
    else:
        image_output_folder = os.path.join(Path(images_input_folder).parent.as_posix(), 'cropped_data/images')
        mask_output_folder = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data/masks')
    crop_size = args.crop_size  # Adjust this according to your needs
    shift = args.step_size  # Adjust this according to your needs


    if args.start_from_scratch:
        if os.path.exists(image_output_folder):
            shutil.rmtree(image_output_folder)
            os.makedirs(image_output_folder)
        else:
            os.makedirs(image_output_folder)
        if os.path.exists(mask_output_folder):
            shutil.rmtree(mask_output_folder)
            os.makedirs(mask_output_folder, exist_ok=True)
        else:
            os.makedirs(mask_output_folder)
    else:
        # Create output folder if it doesn't exist
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)
        if not os.path.exists(mask_output_folder):
            os.makedirs(mask_output_folder)

    # List all image files in the input folder
    images_files = [f for f in os.listdir(images_input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    masks_files = [f for f in os.listdir(masks_input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in tqdm(images_files):
        # Construct the full file paths
        image_input_path = os.path.join(images_input_folder, image_file)
        mask_file = image_file.replace('.', "_mask.")
        masks_input_path = os.path.join(masks_input_folder, mask_file)
        img_output_path = os.path.join(image_output_folder, f"cropped_{image_file}")
        msk_output_path = os.path.join(mask_output_folder, f"cropped_{image_file}")

        # Open the image
        with Image.open(image_input_path) as img:
            # Get the width and height of the image
            width, height = img.size

            with Image.open(masks_input_path) as msk:
                # Get the width and height of the image
                width, height = msk.size

                # Iterate over the image, cropping and saving
                for y in range(0, height, shift):
                    for x in range(0, width, shift):
                        # Crop the image
                        if x + crop_size < IMG_WIDTH and y + crop_size < IMG_HEIGHT:
                            cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
                            cropped_msk = msk.crop((x, y, x + crop_size, y + crop_size))
                        elif x + crop_size > IMG_WIDTH:
                            cropped_img = img.crop((IMG_WIDTH - crop_size, y, IMG_WIDTH, y + crop_size))
                            cropped_msk = msk.crop((IMG_WIDTH - crop_size, y, IMG_WIDTH, y + crop_size))
                        elif y + crop_size > IMG_HEIGHT:
                            cropped_img = img.crop((x, IMG_HEIGHT - crop_size, x + crop_size, IMG_HEIGHT))
                            cropped_msk = msk.crop((x, IMG_HEIGHT - crop_size, x + crop_size, IMG_HEIGHT))

                        if len(np.array(cropped_msk).shape) > 2 and np.array(cropped_msk).shape[-1] > 3:
                            if np.sum(np.array(cropped_msk)[:,:,:3]) > 1:
                                # Save the cropped image to the output folder
                                cropped_msk.save(msk_output_path.replace('.', '_{}_{}_mask.'.format(x, y)))
                                cropped_img.save(img_output_path.replace('.', '_{}_{}.'.format(x, y)))
                                print(np.unique(np.array(cropped_msk)[:,:,:3]))
                            else:
                                if random.random() >= 1 - save_bkg_perc:
                                    cropped_msk.save(msk_output_path.replace('.', '_{}_{}_mask.'.format(x, y)))
                                    cropped_img.save(img_output_path.replace('.', '_{}_{}.'.format(x, y)))
                        else:
                            if np.sum(np.array(cropped_msk)) > 1:
                                # Save the cropped image to the output folder
                                cropped_msk.save(msk_output_path.replace('.', '_{}_{}_mask.'.format(x, y)))
                                cropped_img.save(img_output_path.replace('.', '_{}_{}.'.format(x, y)))
                                #print(np.unique(np.array(cropped_msk)[:,:,:3]))
                            else:
                                if random.random() >= 1 - save_bkg_perc:
                                    cropped_msk.save(msk_output_path.replace('.', '_{}_{}_mask.'.format(x, y)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")

    parser.add_argument("--images_path", default=data_images_path, help="Path to the input image")
    parser.add_argument("--crop_size", type=int, default=512, help="Patch size for extraction")
    parser.add_argument("--step_size", type=int, default=480, help="Step size for the cropping")
    parser.add_argument("--save_bkg_perc", type=int, default=0.25, help="probability to retain a background image")
    parser.add_argument("--start_from_scratch", type=int, default=1, help="remove all the filtered_images into save_path dir")
    parser.add_argument("--total_background", type=int, default=0,
                        help="remove all the filtered_images into save_path dir")
    args = parser.parse_args()

    # Call the function with provided arguments
    crop_images(args)
