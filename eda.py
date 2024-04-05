import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
from config import *

# A name for the dataset
#name = "fusolage_defects"
dataset_dir = Path(train_images_path).parent.as_posix()

data_path = test_1_data_images_path
labels_path = "model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_1/segformer_processor_decoder_w_1_3_2_2024_03_22_09_36_51/test1/merged_model_results"
#labels_path = renamed_masks_path

#data_path = cropped_tot_bkg_images_path
#labels_path = cropped_tot_bkg_renamed_masks_path


# Create the dataset
dataset = fo.Dataset.from_dir(
    #dataset_dir=dataset_dir,
    data_path=data_path,
    labels_path=labels_path,
    dataset_type=fo.types.ImageSegmentationDirectory,
    #name=name,
)


session = fo.launch_app(dataset)
