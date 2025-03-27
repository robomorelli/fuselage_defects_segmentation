import os.path
import random
import shutil
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import sys
import numpy as np
sys.path.append('..')
from pathlib import Path
from config import *

def main(data_path, n_splits=3, val_ratio=0.15, seed=123, start_from_scratch=1):
    """
    Split images and masks into train, validation, and test1 sets and copy them to the output directory.

    Arguments:
    image_path: Path to the directory containing images.
    mask_path: Path to the directory containing masks.
    output_dir: Path to the output directory where the split dataset will be saved.
    train_ratio: Ratio of data to be used for training.
    val_ratio: Ratio of data to be used for validation.
    seed: Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    random.seed(seed, version=2)

    image_path = os.path.join(data_path, 'images')
    oversampling_file_path = args.oversampling_file

    output_dir = os.path.join(data_path, 'k-fold')
    if args.start_from_scratch:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

    # Get image and mask filenames
    image_files = os.listdir(image_path)
    mask_files = [x.replace('.', '_mask.') for x in image_files]

    cropped_image_files = os.listdir(os.path.join(Path(image_path).parent.as_posix(), 'cropped_data/images'))
    cropped_masks_files = [x.replace('.', '_mask.') for x in cropped_image_files]

    if oversampling_file_path is not None:
        oversamplig_file_names = list(pd.read_excel(oversampling_file_path)['name'].values)
    else:
        oversamplig_file_names = []

    # Check if the number of images and masks are the same
    if len(image_files) != len(mask_files):
        raise ValueError("Number of images and masks do not match.")

    # Combine image and mask filenames
    data = list(zip(image_files, mask_files))
    # Create output directories
    for i in range(n_splits):
        split_dir = os.path.join(output_dir,  f'fold_{i+1}')
        os.makedirs(split_dir, exist_ok=True)

        # Initialize KFold splitter
        train_dict = {}
        val_dict = {}
        test_dict = {}
        train_dict_full_images = {}
        val_dict_full_images = {}
        test_dict_full_images = {}

        if args.n_splits == 1 and args.test_from_txt_file is not None:

            with open(args.test_from_txt_file, 'r') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]

            test_index = []
            for element in lines:
                if element in image_files and element not in oversamplig_file_names:
                    test_index.append(image_files.index(element))
                    if element in oversampling_file_path:
                        print(f'REMOVING THIS FILE {element} from test since it is in oversampling')

            test_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in test_index for crop_fh in
                         cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]
            all_index = np.arange(len(data))
            train_index = [i for i in all_index if i not in test_index]

            train_index, val_index = train_test_split(train_index, test_size=val_ratio, random_state=seed)
            train_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in train_index for crop_fh in
                         cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]
            val_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in val_index for crop_fh in
                         cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]
            val_data_full_images = [data[idx] for idx in val_index]
            train_data_full_images = [data[idx] for idx in train_index]
            test_data_full_images = [data[idx] for idx in test_index]

            train_dict["images"] = [x[0] for x in train_data]
            train_dict["masks"] = [x[1] for x in train_data]
            val_dict["images"] = [x[0] for x in val_data]
            val_dict["masks"] = [x[1] for x in val_data]
            test_dict["images"] = [x[0] for x in test_data]
            test_dict["masks"] = [x[1] for x in test_data]

            train_dict_full_images["images"] = [x[0] for x in train_data_full_images]
            train_dict_full_images["masks"] = [x[1] for x in train_data_full_images]
            val_dict_full_images["images"] = [x[0] for x in val_data_full_images]
            val_dict_full_images["masks"] = [x[1] for x in val_data_full_images]
            test_dict_full_images["images"] = [x[0] for x in test_data_full_images]
            test_dict_full_images["masks"] = [x[1] for x in test_data_full_images]

            # Copy train data to respective fold directories
            train_dir = os.path.join(output_dir, f'fold_{i + 1}', 'train')
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)

            os.makedirs(os.path.join(train_dir), exist_ok=True)
            train_filenames = pd.DataFrame(train_dict)
            train_filenames.to_csv(os.path.join(train_dir, f'cropped_filenames.csv'), index=False)
            train_full_images_filenames = pd.DataFrame(train_dict_full_images)
            train_full_images_filenames.to_csv(os.path.join(train_dir, f'full_size_filenames.csv'), index=False)

            val_dir = os.path.join(output_dir, f'fold_{i + 1}', 'val')
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)

            os.makedirs(os.path.join(val_dir), exist_ok=True)
            val_filenames = pd.DataFrame(val_dict)
            val_filenames.to_csv(os.path.join(val_dir, f'cropped_filenames.csv'), index=False)
            val_full_images_filenames = pd.DataFrame(val_dict_full_images)
            val_full_images_filenames.to_csv(os.path.join(val_dir, f'full_size_filenames.csv'), index=False)

            test_dir = os.path.join(output_dir, f'fold_{i + 1}', 'test')
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

            os.makedirs(os.path.join(test_dir), exist_ok=True)
            test_filenames = pd.DataFrame(test_dict)
            test_filenames.to_csv(os.path.join(test_dir, f'cropped_filenames.csv'), index=False)
            test_full_images_filenames = pd.DataFrame(test_dict_full_images)
            test_full_images_filenames.to_csv(os.path.join(test_dir, f'full_size_filenames.csv'), index=False)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            # Split data using KFold
            if args.test_from_txt_file is not None:
                with open(args.test_from_txt_file, 'r') as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines]

                test_index = []
                for element in lines:
                    if element in image_files and element not in oversamplig_file_names:
                        test_index.append(image_files.index(element))
                        if element in oversampling_file_path:
                            print(f'REMOVING THIS FILE {element} from '
                                  f'test since it is in oversampling')

                test_data = [(crop_fh, crop_fh.replace('.', '_mask.'))
                             for idx in test_index for crop_fh in
                             cropped_image_files if 'cropped_' +
                             data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]

            for i, (train_index, val_test_index) in enumerate(kf.split(data)):
                if args.test_from_txt_file is not None:
                    val_index = val_test_index
                    # train_index, valid_index, train data

                else:
                    test_index = val_test_index
                    train_index, val_index = train_test_split(train_index,
                                                    test_size=val_ratio, random_state=seed)
                    test_data = [(crop_fh, crop_fh.replace('.', '_mask.'))
                                 for idx in test_index for crop_fh in
                                 cropped_image_files if 'cropped_' +
                                 data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]

                    # train_index, valid_index, train data
                train_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in train_index for crop_fh in
                             cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]
                val_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in val_index for crop_fh in
                             cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]

                val_data_full_images = [data[idx] for idx in val_index]
                train_data_full_images = [data[idx] for idx in train_index]
                test_data_full_images = [data[idx] for idx in test_index]

                train_dict["images"] = [x[0] for x in train_data]
                train_dict["masks"] = [x[1] for x in train_data]
                val_dict["images"] = [x[0] for x in val_data]
                val_dict["masks"] = [x[1] for x in val_data]
                test_dict["images"] = [x[0] for x in test_data]
                test_dict["masks"] = [x[1] for x in test_data]

                train_dict_full_images["images"] = [x[0] for x in train_data_full_images]
                train_dict_full_images["masks"] = [x[1] for x in train_data_full_images]
                val_dict_full_images["images"] = [x[0] for x in val_data_full_images]
                val_dict_full_images["masks"] = [x[1] for x in val_data_full_images]
                test_dict_full_images["images"] = [x[0] for x in test_data_full_images]
                test_dict_full_images["masks"] = [x[1] for x in test_data_full_images]

                # Copy train data to respective fold directories
                train_dir = os.path.join(output_dir, f'fold_{i + 1}', 'train')
                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir)

                os.makedirs(os.path.join(train_dir), exist_ok=True)
                train_filenames = pd.DataFrame(train_dict)
                train_filenames.to_csv(os.path.join(train_dir, f'cropped_filenames.csv'), index=False)
                train_full_images_filenames = pd.DataFrame(train_dict_full_images)
                train_full_images_filenames .to_csv(os.path.join(train_dir, f'full_size_filenames.csv'), index=False)

                val_dir = os.path.join(output_dir, f'fold_{i + 1}', 'val')
                if os.path.exists(val_dir):
                    shutil.rmtree(val_dir)

                os.makedirs(os.path.join(val_dir), exist_ok=True)
                val_filenames = pd.DataFrame(val_dict)
                val_filenames.to_csv(os.path.join(val_dir, f'cropped_filenames.csv'), index=False)
                val_full_images_filenames = pd.DataFrame(val_dict_full_images)
                val_full_images_filenames .to_csv(os.path.join(val_dir, f'full_size_filenames.csv'), index=False)


                test_dir = os.path.join(output_dir, f'fold_{i + 1}', 'test')
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)

                os.makedirs(os.path.join(test_dir), exist_ok=True)
                test_filenames = pd.DataFrame(test_dict)
                test_filenames.to_csv(os.path.join(test_dir, f'cropped_filenames.csv'), index=False)
                test_full_images_filenames = pd.DataFrame(test_dict_full_images)
                test_full_images_filenames.to_csv(os.path.join(test_dir, f'full_size_filenames.csv'), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")

    parser.add_argument("--data_path", default=data_path, help="Path to the input image")
    parser.add_argument("--n_splits", type=int, default=3, help="Patch size for extraction")
    parser.add_argument("--val_ratio", type=int, default=0.05, help="Patch size for extraction")
    parser.add_argument("--seed", type=int, default=123, help="Patch size for extraction")
    parser.add_argument("--start_from_scratch", type=int, default=1, help="Patch size for extraction")
    parser.add_argument("--test_from_txt_file", default='../data/test_images.txt', help="Patch size for extraction")
    parser.add_argument("--oversampling_file", type=str, default='./oversampling.xlsx', help="")
    args = parser.parse_args()

    main(args.data_path, args.n_splits, args.val_ratio, args.seed, args.start_from_scratch)
