import os.path
import random
import shutil
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
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
    train_dict_full_images = {}
    val_dict_full_images = {}
    test_dict_full_images = {}
    # Split data using KFold
    for i, (train_index, test_index) in enumerate(kf.split(data)):

        test_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in test_index for crop_fh in
                     cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]
        train_index, val_index = train_test_split(train_index, test_size=val_ratio, random_state=seed)
        train_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in train_index for crop_fh in
                     cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]
        val_data = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in val_index for crop_fh in
                     cropped_image_files if 'cropped_' + data[idx][0].split('.')[0] == '_'.join(crop_fh.split('_')[:-2])]


        #test_data = [(crop_fh.replace('_mask.', '.'), crop_fh) for crop_fh in fh]

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
        #os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        #os.makedirs(os.path.join(train_dir, "masks"), exist_ok=True)
        #train_images_filenames = pd.DataFrame(train_dict["images"])
        #train_masks_filenames = pd.DataFrame(train_dict["masks"])
        #train_images_filenames.to_csv(os.path.join(train_dir, "images", f'cropped_images_filenames.csv'), index=False)
        #train_masks_filenames.to_csv(os.path.join(train_dir, "masks", f'cropped_masks_filenames.csv'), index=False)

        val_dir = os.path.join(output_dir, f'fold_{i + 1}', 'val')
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)

        os.makedirs(os.path.join(val_dir), exist_ok=True)
        val_filenames = pd.DataFrame(val_dict)
        val_filenames.to_csv(os.path.join(val_dir, f'cropped_filenames.csv'), index=False)
        val_full_images_filenames = pd.DataFrame(val_dict_full_images)
        val_full_images_filenames .to_csv(os.path.join(val_dir, f'full_size_filenames.csv'), index=False)
        #os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        #os.makedirs(os.path.join(val_dir, "masks"), exist_ok=True)
        #val_images_filenames = pd.DataFrame(val_dict["images"])
        #val_masks_filenames = pd.DataFrame(val_dict["masks"])
        #val_images_filenames.to_csv(os.path.join(val_dir, "images", f'cropped_images_filenames.csv'), index=False)
        #val_masks_filenames.to_csv(os.path.join(val_dir, "masks", f'cropped_masks_filenames.csv'), index=False)

        test_dir = os.path.join(output_dir, f'fold_{i + 1}', 'test')
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        os.makedirs(os.path.join(test_dir), exist_ok=True)
        test_filenames = pd.DataFrame(test_dict)
        test_filenames.to_csv(os.path.join(test_dir, f'cropped_filenames.csv'), index=False)
        test_full_images_filenames = pd.DataFrame(test_dict_full_images)
        test_full_images_filenames.to_csv(os.path.join(test_dir, f'full_size_filenames.csv'), index=False)
        #os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
        #os.makedirs(os.path.join(test_dir, "masks"), exist_ok=True)
        #test_images_filenames = pd.DataFrame(test_dict["images"])
        #test_masks_filenames = pd.DataFrame(test_dict["masks"])
        #test_images_filenames.to_csv(os.path.join(test_dir, "images", f'cropped_images_filenames.csv'), index=False)
        #test_masks_filenames.to_csv(os.path.join(test_dir, "masks", f'cropped_masks_filenames.csv'), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")

    parser.add_argument("--data_path", default=data_path, help="Path to the input image")
    parser.add_argument("--n_splits", type=int, default=9, help="Patch size for extraction")
    parser.add_argument("--val_ratio", type=int, default=0.15, help="Patch size for extraction")
    parser.add_argument("--seed", type=int, default=123, help="Patch size for extraction")
    parser.add_argument("--start_from_scratch", type=int, default=1, help="Patch size for extraction")
    args = parser.parse_args()

    main(args.data_path, args.n_splits, args.val_ratio, args.seed, args.start_from_scratch)
