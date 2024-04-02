import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
from pathlib import Path
import yaml
sys.path.append('../')
from config import *

def main(args):

    priority_list = args.priority_list
    input_folder = args.multiclass_masks_folder
    folds = os.listdir(input_folder)
    output_folder = args.output_folder
    mapping_dict = args.mapping_dict
    viz_folder = os.path.join(Path(output_folder).parent.as_posix(), 'visualization')
    mask_type = args.type


    if len(mapping_dict) == 0:
        mapping_dict = {f: int(i + 1) for i, f in enumerate(folds)}

    num_classes = len(folds)

    step = 55
    min_value = 255 - (step*num_classes)
    viz_palette = [x for x in range(255, min_value-1, -step)]


    #mapping_dict_viz = {f: int(255 / (i + 1)) for i, f in enumerate(folds)}
    mapping_dict_viz = {f: viz_palette[i] for i, f in enumerate(folds)}
    with open('class_mapping_viz.yaml', 'w') as f:
        yaml.dump(mapping_dict_viz, f)

    folds_masks_names = []
    exclude_labels = args.esxclude_labels
    folds = [f for f in folds if f not in exclude_labels]
    for f in folds:
        folds_masks_names.append(os.listdir(os.path.join(input_folder, f)))


    for ix in range(len(folds) - 1):
        assert folds_masks_names[ix] == folds_masks_names[ix+1]

    masks_names = [file for file in folds_masks_names[0] if file.endswith(".png")]

    if int(priority_list[0]) == 1:
        enable_priority = True
    else:
        enable_priority = False
    priority_list = priority_list[1:]

    priority_list = [x for x in priority_list if x not in exclude_labels]

    # Create the output folder if it doesn't exist
    if args.start_from_scratch:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        if mask_type == 'pixel-wise':
            if os.path.exists(viz_folder):
                shutil.rmtree(viz_folder)
            os.makedirs(viz_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(viz_folder, exist_ok=True)

    with open('class_mapping.yaml', 'w') as f:
        yaml.dump(mapping_dict, f)

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


        if mask_type == 'channel-wise':
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, num_classes), dtype=int)
            mask_viz = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            for i, label in enumerate(folds):
                addend = masks_collector[label][:,:,0:1]
                mask[:, :, i] = np.squeeze(addend)
                addend = masks_collector[label].astype(np.uint8)
                mask_viz = mask_viz + addend * mapping_dict_viz[label]

            # mask = mask / 255.
            np.save(os.path.join(output_folder, png_file.replace(".png", ".npy")), mask)
            plt.imsave(os.path.join(viz_folder, png_file), np.squeeze(mask_viz), cmap='gray')
            print('mask unique value', np.unique(mask))

            #if num_classes == 1 or num_classes==3:
            #    plt.imsave(os.path.join(output_folder, png_file), np.squeeze(mask))

        elif mask_type == 'pixel-wise':
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
            mask_viz = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
            for label in reversed(priority_list):
                channel = masks_collector[label].astype(np.uint8)[:, :, :1]

                #if "00000003_2_mask" in png_file:
                #    print('here')

                #mask_viz = mask_viz + channel * mapping_dict_viz[label]
                #mask = mask + channel * mapping_dict[label]
                mask_viz[channel == 1] = mapping_dict_viz[label]
                mask[channel == 1] = mapping_dict[label]

            #mask = mask / 255.

            #plt.imsave(os.path.join(viz_folder, png_file), np.squeeze(mask_viz), cmap='gray')
            #plt.imsave(os.path.join(output_folder, png_file), np.squeeze(mask))
            cv2.imwrite(os.path.join(viz_folder, png_file), np.squeeze(mask_viz))
            cv2.imwrite(os.path.join(output_folder, png_file), np.squeeze(mask))
            print('mask unique value', np.unique(mask))
            print('mask viz unique value', np.unique(mask_viz))
            if len(np.unique(mask)) > len(list(mapping_dict.values())) + 1:
                raise Exception
        else:
            print('type is not recognized')
            NotImplementedError

        print("Masks saved in", os.path.join(output_folder, png_file))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--multiclass_masks_folder", default=multiclass_masks_path, help="Path to the input image")
    parser.add_argument("--output_folder", default=data_masks_path, help="Path to the input image")
    parser.add_argument("--type", default='pixel-wise', help="[channel-wise, pixel-wise]")
    parser.add_argument("--mapping_dict", default={}, help="Path to the input image")
    parser.add_argument('--priority_list', nargs='+', default=['0', 'Mark', 'Graffio'], help='List of items')
    parser.add_argument('--esxclude_labels', nargs='+', default=['Graffio'], help='List of items')
    parser.add_argument("--start_from_scratch", type=int, default=0, help="remove all the filtered_images into save_path dir")

    args = parser.parse_args()
    main(args)
