import numpy as np
from label_studio_sdk import Client
import cv2
import argparse
from pathlib import Path
from label_studio_converter import brush
import os
import yaml
from config import *

LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "f5462754f59476b82c058eeed9e44f6df4d02ff8"

def main(args):

    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()

    name_mapping = {'Graffio': 2, 'Mark': 1}
    palette_viz = {'Graffio': 200, 'Mark': 145}

    #with open('./preprocessing/class_mapping.yaml', 'r') as yaml_file:
    #    name_mapping = yaml.load(yaml_file, Loader=yaml.FullLoader)

    #with open('./preprocessing/class_mapping_viz.yaml', 'r') as yaml_file:
    #    palette_viz = yaml.load(yaml_file, Loader=yaml.FullLoader)

    mapping = {}
    for k in list(name_mapping.keys()):
        mapping[str(name_mapping[str(k)])] = palette_viz[k]

    try:
        masks_input_folder = os.path.join(Path(args.masks_path).parent.as_posix(), "masks")
        file_names = os.listdir(masks_input_folder)
    except:
        masks_input_folder = args.masks_path
        file_names = os.listdir(masks_input_folder)

    trials = 0
    ''' 
    while True:
        try:
            # Your code that might raise an exception goes here
            # For example:
            project = ls.get_projects()[trials]
            tasks = project.get_tasks()
            assert len(tasks) == len(os.listdir(masks_input_folder))
            break  # If successful, break out of the loop
        except Exception as e:
            trials += 1
            print(f"An error occurred increasing trial number to fix: {e}")
    '''
    project = ls.get_projects()[trials]
    tasks = project.get_tasks()
    for tsk in tasks:
        name = os.path.basename(tsk['storage_filename'])
        if args.masks_suffix:
            mask_name = name.replace('.', '_mask.')
        else:
            mask_name = name
        mask_file = os.path.join(masks_input_folder, mask_name)
        mask = cv2.imread(mask_file)

        if mask is not None:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mask = np.squeeze(mask[:,:,0:1])

            if args.multiclass:
                for v in list(mapping.keys()):
                    mask_one_label = mask.copy()
                    if int(v) != 0:
                        #mask_one_label[mask_one_label == int(v)] = mapping[str(v)]
                        mask_one_label[mask_one_label == int(v)] = 255
                        mask_one_label[mask_one_label != 255] = 0

                        print(np.unique(mask_one_label))

                        rle = brush.mask2rle(mask_one_label)

                        project.create_annotation(
                            task_id=tsk['id'],
                            model_version=None,

                            result=[
                                {
                                    "from_name": "brush_labels_tag",
                                    "to_name": "image",
                                    "type": "brushlabels",
                                    'value': {"format": "rle", "rle": rle, "brushlabels": ['LABEL']},
                                }
                            ],
                        )

            else:
                mask = (mask > 0).astype(np.uint8) * 255  # better to threshold, it reduces output annotation size

                print(np.unique(mask))
                rle = brush.mask2rle(mask)

                project.create_annotation(
                    task_id=tsk['id'],
                    model_version=None,
                    result=[
                        {
                            "from_name": "brush_labels_tag",
                            "to_name": "image",
                            "type": "brushlabels",
                            'value': {"format": "rle", "rle": rle, "brushlabels": ['LABEL']},
                        }
                    ],)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--masks_path", default=feb_mar_apr_may_model_results, help="Path to the input image")
    parser.add_argument("--masks_suffix", default=0, help="Path to the input image")
    parser.add_argument("--multiclass", default=1, help="Path to the input image")

    args = parser.parse_args()
    main(args)
