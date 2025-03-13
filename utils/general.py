import os
import random
import shutil
from pathlib import Path
import yaml
import json
import types
from config import *

# Helper function to read YAML files
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_object(dct):
    return types.SimpleNamespace(**dct)


def read_config(config_name):
    with open(os.path.join(config_name), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)
    return cfg

def sample_and_divide(images_input_folder, sample_percentage=0.3,
                      train_ratio=0.75, divide=True, custom_list = [], start_from_scratch=0):

    masks_input_folder = os.path.join(Path(images_input_folder).parent, 'masks')
    # Check if output folder exists, create if not

    if start_from_scratch:
        if not os.path.exists(fine_tuning_train_images_path):
            os.makedirs(fine_tuning_train_images_path)
        else:
            shutil.rmtree(fine_tuning_train_images_path)
            os.makedirs(fine_tuning_train_images_path)
        if not os.path.exists(fine_tuning_train_masks_path):
            os.makedirs(fine_tuning_train_masks_path)
        else:
            shutil.rmtree(fine_tuning_train_masks_path)
            os.makedirs(fine_tuning_train_masks_path)

        if not os.path.exists(fine_tuning_val_images_path):
            os.makedirs(fine_tuning_val_images_path)
        else:
            shutil.rmtree(fine_tuning_val_images_path)
            os.makedirs(fine_tuning_val_images_path)
        if not os.path.exists(fine_tuning_val_masks_path):
            os.makedirs(fine_tuning_val_masks_path)
        else:
            shutil.rmtree(fine_tuning_val_masks_path)
            os.makedirs(fine_tuning_val_masks_path)
    else:
        if not os.path.exists(fine_tuning_train_images_path):
            os.makedirs(fine_tuning_train_images_path)
        if not os.path.exists(fine_tuning_train_masks_path):
            os.makedirs(fine_tuning_train_masks_path)
        if not os.path.exists(fine_tuning_val_images_path):
            os.makedirs(fine_tuning_val_images_path)
        if not os.path.exists(fine_tuning_val_masks_path):
            os.makedirs(fine_tuning_val_masks_path)

    # Get list of all files in input folder
    if len(custom_list) == 0:
        images = [f for f in os.listdir(images_input_folder) if os.path.isfile(os.path.join(images_input_folder, f))]
    else:
        images = [f for f in custom_list if os.path.isfile(os.path.join(images_input_folder, f))]

    if divide:
        # Calculate the number of samples needed
        num_samples = int(len(images) * sample_percentage)

        # Sample images randomly without replacement
        sampled_images = random.sample(images, num_samples)

        # Calculate number of images for train and validation
        num_train = int(len(sampled_images) * train_ratio)
        num_val = len(sampled_images) - num_train

        # Randomly split sampled images into train and validation sets
        train_images = random.sample(sampled_images, num_train)
        val_images = [img for img in sampled_images if img not in train_images]

        # Copy train images to the train folder
        for image in train_images:
            shutil.copy(os.path.join(images_input_folder, image), os.path.join(fine_tuning_train_images_path, image))
            shutil.copy(os.path.join(masks_input_folder, image.replace('.', '_mask.'))
                        , os.path.join(fine_tuning_train_masks_path, image.replace('.', '_mask.')))

        # Copy validation images to the validation folder
        for image in val_images:
            shutil.copy(os.path.join(images_input_folder, image), os.path.join(fine_tuning_val_images_path, image))
            shutil.copy(os.path.join(masks_input_folder, image.replace('.', '_mask.'))
                        , os.path.join(fine_tuning_val_masks_path, image.replace('.', '_mask.')))
    else:
        # Copy train images to the train folder
        for image in images:
            shutil.copy(os.path.join(images_input_folder, image), os.path.join(fine_tuning_train_images_path, image))
            shutil.copy(os.path.join(masks_input_folder, image.replace('.', '_mask.'))
                        , os.path.join(fine_tuning_train_masks_path, image.replace('.', '_mask.')))


if __name__ == '__main__':

    images_input_folder = data_images_path
    divide = False
    start_from_scratch = 0
    sample_percentage = 0.3
    train_ratio = 0.75

    #custom_list = []
    custom_list = custom_list

    sample_and_divide(images_input_folder=images_input_folder,
                      sample_percentage=sample_percentage, train_ratio=train_ratio, divide=divide,
                      custom_list = custom_list, start_from_scratch=start_from_scratch)
