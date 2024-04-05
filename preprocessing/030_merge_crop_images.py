import os.path
import shutil
import numpy as np
import cv2
import argparse
from pathlib import Path
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure
from config import *


def main(args):

    input_folder = args.crops_path
    #if args.use_df_filename:
    #    fold = os.path.basename(Path(input_folder).parent.parent).split('_')[1]
    #    df_path = os.path.join(k_fold_data_path, f"fold_{fold}", args.split)
    #    df_names = pd.read_csv(os.path.join(df_path, "full_size_filenames.csv"))
    #    images_file_names = df_names['images']
    #    masks_file_names = df_names['masks']

    full_size_images_path = args.full_size_images_path
    full_size_masks_path = os.path.join(Path(args.full_size_images_path).parent.as_posix(), 'masks')
    suffix = os.path.basename(input_folder)
    output_folder = os.path.join(Path(args.crops_path).parent.as_posix(), f"merged_{suffix}")
    output_folder_viz = os.path.join(Path(args.crops_path).parent.as_posix(), f"merged_{suffix}_viz")
    output_folder_contours = os.path.join(Path(args.crops_path).parent.as_posix(), f"merged_{suffix}_contours")

    fold = os.path.basename(Path(input_folder).parent.parent).split('_')[1]

    num_classes = len(os.listdir(full_size_masks_classes_path))
    print('multiclass for n classes', num_classes)
    classes_name = os.listdir(full_size_masks_classes_path)
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
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    if args.start_from_scratch:
        if os.path.exists(output_folder_viz):
            shutil.rmtree(output_folder_viz)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_viz, exist_ok=True)

    crop_filenames = os.listdir(input_folder)
    if "mask" in crop_filenames[0]:
        crop_filenames = [x.replace('_mask.', '.') for x in crop_filenames]
        mask = True
    else:
        mask = False

    crop_filenames = [x.replace('.png', '') for x in crop_filenames]
    crop_basename = ['_'.join(cfh.split('_')[:-2]) for cfh in crop_filenames]
    crop_basename = list(np.unique(crop_basename))

    if args.plot_contours:
        if args.start_from_scratch:
            if os.path.exists(output_folder_contours):
                shutil.rmtree(output_folder_contours)
        os.makedirs(output_folder_contours, exist_ok=True)

        id_classes = [i + 1 for i in range(num_classes)]

    for basename in crop_basename:
        #basename = '_'.join(cfh.split('_')[:-2])
        crop_to_merge = [crop_fh for crop_fh in crop_filenames if '_'.join(crop_fh.split('_')[:-2]) == basename]

        x_splits = [int(crop_fh.split('_')[-2]) for crop_fh in crop_to_merge]
        y_splits = [int(crop_fh.split('_')[-1]) for crop_fh in crop_to_merge]

        x_splits = np.unique(x_splits)
        y_splits = np.unique(y_splits)

        x_splits.sort()
        y_splits.sort()

        image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=int)

        for ixs, x_s in enumerate(x_splits[:-1]):
            for iys, y_s in enumerate(y_splits[:-1]):
                if mask:
                    crop_path = os.path.join(input_folder, (basename + f"_{x_s}" + f"_{y_s}_mask.png"))
                else:
                    crop_path = os.path.join(input_folder, (basename + f"_{x_s}" + f"_{y_s}.png"))
                crop = cv2.imread(crop_path)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)[:,:,0:1]
                image[y_s: y_splits[iys + 1], x_s: x_splits[ixs + 1]] = crop

        if args.plot_contours:

            kernel = np.ones((15, 15), np.uint8)  # Adjust the kernel size as needed

            img = cv2.imread(os.path.join(full_size_images_path, f"{basename.lstrip('cropped_')}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(os.path.join(full_size_masks_path, f"{basename.lstrip('cropped_')}.png".replace('.','_mask.')))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)[:,:,0:1]

            for class_value in id_classes:
                # Create a mask for the current class value
                class_mask = np.uint8(image == class_value)
                # Perform dilation
                class_mask= cv2.dilate(class_mask, kernel, iterations=1)

                class_gt = np.uint8(gt == class_value)
                # Perform dilation
                class_gt = cv2.dilate(class_gt, kernel, iterations=2)

                # Find contours
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_mask, _ = cv2.findContours(class_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw contours with different colors depending on the class value
                color = (0, 255, 0) if class_value == 1 else (255, 0, 0)
                cv2.drawContours(img, contours, -1, color, thickness=2)
                color = (0, 128, 0) if class_value == 1 else (255, 255, 0)
                cv2.drawContours(img, contours_mask, -1, color, thickness=2)
                legend = np.zeros((150, IMG_WIDTH, 3), dtype=np.uint8)
                cv2.putText(legend, 'Mark: Green', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 4)
                cv2.putText(legend, 'Graffio : Blu', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 4)
                cv2.putText(legend, 'Mark_gt: Olive', (1000, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 128, 0), 4)
                cv2.putText(legend, 'Graffio_gt : Cyan', (1000, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 0), 4)
                #text_positions = [(10, i * 20 + 60) for i in range(len(classes_name))]  # Calculate text positions
                #[img := cv2.putText(legend, f'{classes_name[i]}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color,
                #                           4) for i, (text_pos, color) in enumerate(zip(text_positions, colors))]

                combined_image = np.vstack((img, legend))
            cv2
            cv2.imwrite(os.path.join(output_folder_contours, f"{basename.lstrip('cropped_')}.png"), combined_image)


        cv2.imwrite(os.path.join(output_folder, f"{basename.lstrip('cropped_')}.png"), np.squeeze(image))
        print('before',np.unique(image))
        image = (np.array(image) / num_classes)*255
        print('after',np.unique(image))
        cv2.imwrite(os.path.join(output_folder_viz, f"{basename.lstrip('cropped_')}.png"), np.squeeze(image))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    #parser.add_argument("--crops_path", default=os.path.join(common_path_test_results, f"model_results_{0.45}"), help="")
    parser.add_argument("--crops_path", default="../model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_1/segformer_processor_decoder_w_1_3_2_2024_03_22_09_36_51/test/model_results_0.3",
                        help="")
    parser.add_argument("--full_size_images_path", default=data_images_path,
                        help="")
    #parser.add_argument("--use_df_filename", default=1,
    #                    help="")
    #parser.add_argument("--split", default="test",
    #                    help="")
    parser.add_argument("--start_from_scratch", default=1, help="")
    parser.add_argument("--plot_contours", default=1, help="")

    args = parser.parse_args()
    main(args)
