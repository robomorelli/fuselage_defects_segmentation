import os.path
import shutil
import numpy as np
import cv2
import argparse
import pandas as pd
from config import *
import json
from datetime import datetime
from label_studio_sdk import Client

def key_function(key):
    return datetime.strptime(key[:-1], '%Y-%m-%dT%H:%M:%S.%f')

def main(args):

    masks_input_folder = args.pre_masks_path
    masks_output_folder = args.refined_masks_path
    masks_names = os.listdir(masks_input_folder)

    if args.start_from_scratch:
        if os.path.exists(masks_output_folder):
            shutil.rmtree(masks_output_folder)

    os.makedirs(masks_output_folder, exist_ok=True)

    # Opening JSON file
    f = open(f'{args.name_association_file}.json')

    data = json.load(f)

    for ix, d in enumerate(data):
        name = os.path.basename(d['data']['image'])
        mask_output_name = name.replace(".png", "_mask.png")

        id = d["id"]
        mask_input_name_candidates = [x for x in masks_names if str(id) in x]

        if len(mask_input_name_candidates) == 0:
            mask = np.zeros((IMG_HEIGHT, IMG_HEIGHT, 1), dtype=np.uint8)
            cv2.imwrite(os.path.join(masks_output_folder, mask_output_name), mask)

        elif len(mask_input_name_candidates) > 1:
            dict_updated = {}
            for ix, d in enumerate(data):
                if d['id'] == id:
                    dict_updated[d['updated_at']] = d['annotations'][0]['id']

            # Use the max function to find the key with the maximum timestamp
            most_updated_key = max(dict_updated.keys(), key=key_function)
            id_up = str(dict_updated[most_updated_key])
            mask_input_name_candidates = [x for x in mask_input_name_candidates if id_up in x]
            print('more than one association', mask_input_name_candidates)
            mask_input_name_candidates = [x for x in mask_input_name_candidates if 'Graffio' in x]
            shutil.copy(os.path.join(masks_input_folder, mask_input_name_candidates[0]), os.path.join(masks_output_folder, mask_output_name))

        else:
            shutil.copy(os.path.join(masks_input_folder, mask_input_name_candidates[0]), os.path.join(masks_output_folder, mask_output_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--pre_masks_path", default=to_rename_masks_path, help="Path to the input image")
    parser.add_argument("--refined_masks_path", default=data_masks_path, help="Path to the input image")
    parser.add_argument("--name_association_file", default="./data/name_association", help="Path to the input image")
    parser.add_argument("--start_from_scratch", default=1, help="Path to the input image")

    args = parser.parse_args()
    main(args)

    # Separatamente task Mark e Graffio:
    # Mettere maschere label graffio nel nome tutt ein una cartella il cui path coincide con "refined_masks_path" (veri args)
    #
