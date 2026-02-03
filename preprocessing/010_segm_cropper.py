import os.path
import numpy as np
import sys
sys.path.append('..')
from PIL import Image
import argparse
import shutil
import random
from tqdm import tqdm
from pathlib import Path
import cv2
import pandas as pd
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
    oversampling_file_path = args.oversampling_file
    images_input_folder = args.images_path
    masks_input_folder = os.path.join(Path(images_input_folder).parent.as_posix(), 'masks')
    save_bkg_perc = args.save_bkg_perc
    if args.total_background:
        image_output_folder = os.path.join(Path(images_input_folder).parent.as_posix(), 'cropped_data/tot_bkg/images')
        mask_output_folder = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data/tot_bkg/masks')
        mask_output_folder_viz = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data_viz/tot_bkg/masks')
        mask_output_folder_bboxes_viz = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data_bboxes_viz/tot_bkg/masks')
        save_bkg_perc = 1.00
    else:
        image_output_folder = os.path.join(Path(images_input_folder).parent.as_posix(), 'cropped_data/images')
        mask_output_folder = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data/masks')
        mask_output_folder_viz = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data_viz/masks')
        mask_output_folder_bboxes_viz = os.path.join(Path(masks_input_folder).parent.as_posix(), 'cropped_data_bboxes_viz/masks')
    crop_size = args.crop_size  # Adjust this according to your needs
    shift = args.step_size  # Adjust this according to your needs

    if args.start_from_scratch and not args.use_done_list:
        if os.path.exists(image_output_folder):
            shutil.rmtree(image_output_folder)
            os.makedirs(image_output_folder)
        else:
            os.makedirs(image_output_folder)

        if os.path.exists(mask_output_folder):
            shutil.rmtree(mask_output_folder)
            os.makedirs(mask_output_folder)
        else:
            os.makedirs(mask_output_folder)

        if os.path.exists(mask_output_folder_viz):
            shutil.rmtree(mask_output_folder_viz)
            os.makedirs(mask_output_folder_viz)
        else:
            os.makedirs(mask_output_folder_viz)
    else:
        os.makedirs(image_output_folder, exist_ok=True)
        os.makedirs(mask_output_folder, exist_ok=True)
        os.makedirs(mask_output_folder_viz, exist_ok=True)

        ''' 
        if os.path.exists(mask_output_folder_bboxes_viz):
            shutil.rmtree(mask_output_folder_bboxes_viz)
            os.makedirs(mask_output_folder_bboxes_viz)
        else:
            os.makedirs(mask_output_folder_bboxes_viz)
        '''

    # List all image files in the input folder
    images_files = [f for f in os.listdir(images_input_folder)
                    if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if oversampling_file_path is not None:
        oversamplig_file_names = [x.strip(' ') for x in list(pd.read_excel(oversampling_file_path)['name'].values)]
        oversamplig_factor = list(pd.read_excel(oversampling_file_path)['factor'].values)
        oversampling_dict = {k:v*args.base_oversampling_factor for k, v in zip(oversamplig_file_names, oversamplig_factor)}
    else:
        oversamplig_file_names = []
        oversampling_dict = {}

    if args.convert_from_npy:
        if not os.path.exists(masks_input_folder):
            os.makedirs(masks_input_folder)
        masks_files = [f for f in os.listdir(data_masks_npy_path) if
                       f.endswith(('.npy'))]
        for mf in tqdm(masks_files):
            masks_input_path = os.path.join(data_masks_npy_path, mf)
            npy_data = np.load(masks_input_path)

            if npy_data.max() > 255 or npy_data.min() < 0:
                npy_data = 255 * (npy_data - npy_data.min()) / (
                        npy_data.max() - npy_data.min())

            npy_data_uint8 = npy_data.astype(np.uint8)
            image = Image.fromarray(npy_data_uint8)
            image.save(os.path.join(masks_input_folder, mf.replace('.npy', '_mask.png')))

    done = []
    if args.use_done_list:
        done_file_path = os.path.join(Path(image_output_folder).parent.as_posix(), "done_cropped.txt")
        with open(done_file_path, "w") as f:
            for line in f:
                done.append(line.strip())

    for image_file in tqdm(images_files):
        # Construct the full file paths
        image_input_path = os.path.join(images_input_folder, image_file)
        mask_file = image_file.replace('.', "_mask.")
        masks_input_path = os.path.join(masks_input_folder, mask_file)
        img_output_path = os.path.join(image_output_folder, f"cropped_{image_file}")
        msk_output_path = os.path.join(mask_output_folder, f"cropped_{image_file}")
        msk_output_path_viz = os.path.join(mask_output_folder_viz, f"cropped_{image_file}")
        #bboxes_output_path_viz = os.path.join(mask_output_folder_bboxes_viz, f"cropped_{image_file}")
        done.append(image_file)

        try:
            # Open the image
            with Image.open(image_input_path) as img:
                # Get the width and height of the image
                width, height = img.size
                img = img.convert('RGB')

                try:
                    with Image.open(masks_input_path) as msk:
                        # Get the width and height of the image
                        width, height = msk.size
                        msk = msk.convert('L')

                except:
                    width, height = img.size
                    msk = Image.new('RGB', (width, height), (0, 0, 0))
                    msk = msk.convert('L')
                    print(f'no mask for {mask_file}')

                '''
                if np.any(np.unique(msk)) > num_classes:
                    #print(np.unique(msk))
                    print(f"value over classes number {masks_input_path} - {np.unique(msk)}")
                else:
                    print(np.unique(msk))
                    continue
                '''

                # Iterate over the image, cropping and saving
                for y in range(0, height, shift):
                    for x in range(0, width, shift):

                        if os.path.basename(img_output_path.replace('.', '_{}_{}.'.format(x, y))) in oversamplig_file_names:
                            oversamplig_factor = oversampling_dict[os.path.basename(img_output_path.replace('.', '_{}_{}.'.format(x, y)))]
                            print(os.path.basename(img_output_path.replace('.', '_{}_{}.'.format(x, y))))
                        else:
                            oversamplig_factor = 1
                        # Crop the image
                        if x + crop_size < IMG_WIDTH and y + crop_size < IMG_HEIGHT:
                            cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
                            cropped_msk = msk.crop((x, y, x + crop_size, y + crop_size))
                        elif x + crop_size >= IMG_WIDTH:
                            cropped_img = img.crop((IMG_WIDTH - crop_size, y, IMG_WIDTH, y + crop_size))
                            cropped_msk = msk.crop((IMG_WIDTH - crop_size, y, IMG_WIDTH, y + crop_size))
                        elif y + crop_size >= IMG_HEIGHT:
                            cropped_img = img.crop((x, IMG_HEIGHT - crop_size, x + crop_size, IMG_HEIGHT))
                            cropped_msk = msk.crop((x, IMG_HEIGHT - crop_size, x + crop_size, IMG_HEIGHT))

                        if np.sum(np.array(cropped_msk)) > 1:
                            #print(f'label on {masks_input_path}')
                            # Save the cropped image to the output folder
                            for j in range(oversamplig_factor):

                                cropped_img.save(img_output_path.replace('.', '_{}_{}.'.format(x+j, y+j)))
                                cropped_msk = np.array(cropped_msk)

                                if cropped_msk.max() > num_classes:
                                    print('before',np.unique(cropped_msk))
                                    for i in range(num_classes):
                                        cropped_msk[cropped_msk == 255 / (i + 1)] = num_classes - i
                                    print(f'after {mask_file}', np.unique(cropped_msk))
                                    if np.any(np.unique(cropped_msk)) > num_classes:
                                        raise Exception(f"value over classes number {masks_input_path} - {np.unique(msk)}")
                                print('np unique', np.unique(cropped_msk))
                                cv2.imwrite(msk_output_path.replace('.', '_{}_{}_mask.'.format(x+j, y+j)),
                                            np.squeeze(cropped_msk))
                                #print(np.unique(cropped_msk))
                                cropped_msk_viz = (np.array(cropped_msk)/num_classes)*255
                                cv2.imwrite(msk_output_path_viz.replace('.', '_{}_{}_mask.'.format(x+j, y+j)),
                                            np.squeeze(cropped_msk_viz))
                                ### SAVE BBOXES OF ANNOTATED MASK

                        else:
                            #print(f'no label on {masks_input_path}')
                            if random.random() >= 1 - save_bkg_perc:
                                cropped_msk = np.array(cropped_msk)
                                if cropped_msk.max() > num_classes:
                                    #print('before', np.unique(cropped_msk))
                                    for i in range(num_classes):
                                        cropped_msk[cropped_msk == 255 / (i + 1)] = num_classes - i
                                    #print('after', np.unique(cropped_msk))
                                    if np.any(np.unique(cropped_msk)) > num_classes:
                                        raise Exception(f"value over classes number {masks_input_path} - {np.unique(msk)}")
                                print('np unique', np.unique(cropped_msk))
                                cv2.imwrite(msk_output_path.replace('.', '_{}_{}_mask.'.format(x, y))
                                            , np.squeeze(np.array(cropped_msk)))
                                cropped_img.save(img_output_path.replace('.', '_{}_{}.'.format(x, y)))
                                cropped_msk_viz = (np.array(cropped_msk) / num_classes)*255
                                cv2.imwrite(msk_output_path_viz.replace('.', '_{}_{}_mask.'.format(x, y)),
                                            np.squeeze(cropped_msk_viz))


        except:
            # Save missing images to a text file
            if len(done) > 0:
                done_file_path = os.path.join(Path(mask_output_folder).parent.as_posix(), "done_cropped.txt")
                with open(done_file_path, "w") as f:
                    for image_name in done:
                        f.write(image_name + "\n")

                print(f"Missing image names saved to {done_file_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--images_path", default=data_images_path, help="Path to the input image")
    parser.add_argument("--crop_size", type=int, default=512, help="Patch size for extraction - 512")
    parser.add_argument("--step_size", type=int, default=480, help="Step size for the cropping - 480")
    parser.add_argument("--save_bkg_perc", type=int, default=0.20, help="probability to retain a background image - 0.08")
    parser.add_argument("--start_from_scratch", type=int, default=1, help="remove all the filtered_images into save_path dir")
    parser.add_argument("--convert_from_npy", type=int, default=0, help="remove all the filtered_images into save_path dir")
    parser.add_argument("--total_background", type=int, default=0, help="remove all the filtered_images into save_path dir")
    parser.add_argument("--oversampling_file", type=str, default='./oversampling.xlsx', help="")
    parser.add_argument("--base_oversampling_factor", type=int, default=300, help="remove all the filtered_images into save_path dir")
    parser.add_argument("--use_done_list", type=int, default=0, help="remove all the filtered_images into save_path dir")
    args = parser.parse_args()


    # 512 - 480 step- 0.06-0.08 - perc bkg

    # Call the function with provided arguments
    crop_images(args)
