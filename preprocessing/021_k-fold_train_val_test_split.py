import os
import random
import shutil
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from config import *

def main(data_path, n_splits=3, val_ratio=0.15, seed=123):
    """
    Split images and masks into train, validation, and test sets and copy them to the output directory.

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
    mask_path = os.path.join(data_path, 'masks')
    output_dir = os.path.join(data_path, 'k-fold')

    # Get image and mask filenames
    image_files = os.listdir(image_path)
    mask_files = [x.replace('.', '_mask.') for x in image_files]

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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_dict = {}
    val_dict = {}
    test_dict = {}
    # Split data using KFold
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        test_data = [data[idx] for idx in test_index]
        train_index, val_index = train_test_split(train_index, test_size=val_ratio, random_state=seed)
        val_data = [data[idx] for idx in val_index]
        train_data = [data[idx] for idx in test_index]

        train_dict["images"] = [x[0] for x in train_data]
        train_dict["masks"] = [x[1] for x in train_data]
        val_dict["images"] = [x[0] for x in val_data]
        val_dict["masks"] = [x[1] for x in val_data]
        test_dict["images"] = [x[0] for x in test_data]
        test_dict["masks"] = [x[1] for x in test_data]

        # Copy train data to respective fold directories
        train_dir = os.path.join(output_dir, f'fold_{i + 1}', 'train')
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)

        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "masks"), exist_ok=True)
        for image_file, mask_file in train_data:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(train_dir, "images", image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(train_dir, "masks", mask_file))


        val_dir = os.path.join(output_dir, f'fold_{i + 1}', 'val')
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "masks"), exist_ok=True)
        for image_file, mask_file in val_data:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(val_dir, "images", image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(val_dir, "masks", mask_file))

        # Copy test data to respective fold directories
        test_dir = os.path.join(output_dir, f'fold_{i + 1}', 'test')
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "masks"), exist_ok=True)
        for image_file, mask_file in test_data:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(test_dir, "images", image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(test_dir, "masks", mask_file))

        train_filenames_df = pd.DataFrame(train_dict)
        val_filenames_df = pd.DataFrame(val_dict)
        test_filenames_df = pd.DataFrame(test_dict)

        train_filenames_df.to_csv(os.path.join(output_dir, f'train_filenames_fold_{i}.csv'), index=False)
        val_filenames_df.to_csv(os.path.join(output_dir, f'val_filenames_fold_{i}.csv'), index=False)
        test_filenames_df.to_csv(os.path.join(output_dir, f'test_filenames_fold_{i}.csv'), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")

    parser.add_argument("--data_path", default=data_path, help="Path to the input image")
    parser.add_argument("--n_splits", type=int, default=3, help="Patch size for extraction")
    parser.add_argument("--val_ratio", type=int, default=0.15, help="Patch size for extraction")
    parser.add_argument("--seed", type=int, default=123, help="Patch size for extraction")
    args = parser.parse_args()

    main(args.data_path, args.n_splits, args.val_ratio, args.seed)
