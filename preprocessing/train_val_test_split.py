import os
import random
import shutil
import argparse
from config import *

def main(data_path, train_ratio=0.75, val_ratio=0.15, seed=123):
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
    random.seed(seed)

    image_path = os.path.join(data_path, 'images')
    mask_path = os.path.join(data_path, 'masks')

    # Create output directories
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')
    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(os.path.join(directory, 'images'))
            os.makedirs(os.path.join(directory, 'masks'))
        else:
            os.makedirs(os.path.join(directory, 'images'))
            os.makedirs(os.path.join(directory, 'masks'))
    # Get image and mask filenames
    image_files = os.listdir(image_path)
    mask_files = [x.replace('.', '_mask.') for x in image_files]

    train_images_dir = os.path.join(train_dir, 'images')
    train_masks_dir = os.path.join(train_dir, 'masks')
    val_images_dir = os.path.join(val_dir, 'images')
    val_masks_dir = os.path.join(val_dir, 'masks')

    test_images_dir = os.path.join(test_dir, 'images')
    test_masks_dir = os.path.join(test_dir, 'masks')

    # Check if the number of images and masks are the same
    if len(image_files) != len(mask_files):
        raise ValueError("Number of images and masks do not match.")

    # Combine image and mask filenames
    data = list(zip(image_files, mask_files))

    # Shuffle the data
    random.shuffle(data)

    # Calculate split sizes
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    # Copy data to respective directories
    for i, (image_file, mask_file) in enumerate(data):
        if i < train_size:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(train_images_dir, image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(train_masks_dir, mask_file))
        elif i < train_size + val_size:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(val_images_dir, image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(val_masks_dir, mask_file))
        else:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(test_images_dir, image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(test_masks_dir, mask_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")

    parser.add_argument("--data_path", default=data_path, help="Path to the input image")
    parser.add_argument("--train_split", type=int, default=0.75, help="Patch size for extraction")
    parser.add_argument("--val_split", type=int, default=0.15, help="Patch size for extraction")
    parser.add_argument("--seed", type=int, default=123, help="Patch size for extraction")
    args = parser.parse_args()

    main(args.data_path, args.train_split, args.val_split, args.seed)
