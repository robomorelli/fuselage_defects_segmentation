import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
sys.path.append('../')
from config import *

def main(args):

    priority_list = args.priority_list
    input_folder = args.multiclass_masks_folder
    folds = os.listdir(input_folder)
    output_folder = args.output_folder
    mapping_dict = args.mapping_dict
    type = args.type

    num_classes = len(folds)

    folds_masks_names = []
    for f in folds:
        folds_masks_names.append(os.listdir(os.path.join(input_folder,f)))

    if len(mapping_dict) == 0:
        mapping_dict = {f: int(255/(i+1)) for i, f in enumerate(folds)}

    for ix in range(len(folds) - 1):
        assert folds_masks_names[ix] == folds_masks_names[ix+1]

    masks_names = [file for file in folds_masks_names[0] if file.endswith(".png")]

    if int(priority_list[0]) == 1:
        enable_priority = True
    else:
        enable_priority = False
    priority_list = priority_list[1:]

    # Create the output folder if it doesn't exist
    if args.start_from_scratch:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for png_file in masks_names:
        if png_file in os.listdir(output_folder):
            continue
        masks_collector = {}
        for f in folds:
            # load the image
            img_file = os.path.join(input_folder, f, png_file)
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            threshold = 100
            image = np.where(image > threshold, 1, 0)
            if image.shape[0] != IMG_HEIGHT or image.shape[1] != IMG_WIDTH:
                print('shape mismatch')
                if np.sum(image) == 0:
                    image = np.zeros((IMG_HEIGHT, IMG_WIDTH, image.shape[-1]), dtype=int)
                else:
                    raise Exception('size mismatch')

            masks_collector[f] = image

        if enable_priority:
            for ix, label_min in enumerate(reversed(priority_list[1:])):
                img_minuend = masks_collector[label_min]
                for label in priority_list[:-(ix + 1)]:
                    img_subtracting = masks_collector[label]

                    # Create masks for pixels greater than zero in both images
                    minuend_mask = img_minuend > 0
                    subtracting_mask = img_subtracting > 0

                    # Subtract the images where both pixels are greater than zero
                    if minuend_mask.shape != subtracting_mask.shape:
                        print(minuend_mask.shape, subtracting_mask.shape)

                    result = np.where(minuend_mask & subtracting_mask, 1, 0)
                    img_minuend = img_minuend - result

                masks_collector[label_min] = img_minuend

        if type == 'channel-wise':
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, num_classes), dtype=int)
            for i, label in enumerate(priority_list):
                addend = masks_collector[label][:,:,0:1]
                mask[:, :, i] = np.squeeze(addend)

            # mask = mask / 255.
            np.save(os.path.join(output_folder, png_file.replace(".png", ".npy")), mask)

            #if num_classes == 1 or num_classes==3:
            #    plt.imsave(os.path.join(output_folder, png_file), np.squeeze(mask))

        elif type == 'pixel-wise':
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
            for label in priority_list:
                addend = masks_collector[label]
                mask = mask + addend * mapping_dict[label]

            #mask = mask / 255.
            plt.imsave(os.path.join(output_folder, png_file), np.squeeze(mask), cmap='gray')
        else:
            print('type is not recognized')
            NotImplementedError

        print("Masks saved in", output_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--multiclass_masks_folder", default=multiclass_masks_path, help="Path to the input image")
    parser.add_argument("--output_folder", default=data_masks_path, help="Path to the input image")
    parser.add_argument("--type", default='channel-wise', help="[channel-wise, pixel-wise]")
    parser.add_argument("--mapping_dict", default={}, help="Path to the input image")
    parser.add_argument('--priority_list', nargs='+', default=['1', 'Mark', 'Graffio'], help='List of items')
    parser.add_argument("--start_from_scratch", type=int, default=1, help="remove all the filtered_images into save_path dir")

    args = parser.parse_args()
    main(args)
