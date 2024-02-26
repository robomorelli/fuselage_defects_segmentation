import numpy as np
from label_studio_sdk import Client
import cv2
import argparse
from pathlib import Path
from label_studio_converter import brush
from config import *

LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "f5462754f59476b82c058eeed9e44f6df4d02ff8"

def main(args):

    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()

    project = ls.get_projects()[0]

    masks_input_folder = os.path.join(Path(args.masks_path).parent.as_posix(), "masks")

    tasks = project.get_tasks()

    assert len(tasks) == len(os.listdir(masks_input_folder))

    for tsk in tasks:
        name = os.path.basename(tsk['storage_filename'])
        mask_name = name.replace('.', '_mask.')
        mask_file = os.path.join(masks_input_folder, mask_name)
        mask = cv2.imread(mask_file)

        if mask is not None:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mask = np.squeeze(mask[:,:,0:1])
            mask = (mask > 128).astype(np.uint8) * 255  # better to threshold, it reduces output annotation size
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
    parser.add_argument("--masks_path", default=data_masks_path, help="Path to the input image")

    args = parser.parse_args()
    main(args)
