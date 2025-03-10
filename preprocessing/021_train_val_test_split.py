import random
import shutil
import argparse
from sklearn.model_selection import train_test_split
from config import *

def main(data_path, train_ratio=0.75, val_ratio=0.15, seed=123):
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

    # Check if the number of images and masks are the same
    if len(image_files) != len(mask_files):
        raise ValueError("Number of images and masks do not match.")

    # Combine image and mask filenames
    data = list(zip(image_files, mask_files))

    # Shuffle the data
    #random.shuffle(data)
    test_ratio = 1 - train_ratio - val_ratio
    remaining_data, test_data = train_test_split(data, test_size=test_ratio, random_state=seed)
    train_data, val_data = train_test_split(remaining_data, test_size=val_ratio / (1 - test_ratio), random_state=seed)

    # Copy data to respective directories
    for dataset, directory in [(train_data, train_dir), (val_data, val_dir), (test_data, test_dir)]:
        for image_file, mask_file in dataset:
            shutil.copy(os.path.join(image_path, image_file), os.path.join(os.path.join(directory, 'images'), image_file))
            shutil.copy(os.path.join(mask_path, mask_file), os.path.join(os.path.join(directory, 'masks'), mask_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop image and update annotation")

    parser.add_argument("--data_path", default=data_path, help="Path to the input image")
    parser.add_argument("--train_split", type=int, default=0.75, help="Patch size for extraction")
    parser.add_argument("--val_split", type=int, default=0.15, help="Patch size for extraction")
    parser.add_argument("--seed", type=int, default=123, help="Patch size for extraction")
    args = parser.parse_args()

    main(args.data_path, args.train_split, args.val_split, args.seed)
