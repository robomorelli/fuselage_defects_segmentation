import fiftyone as fo
from config import *
import argparse
import cv2

def get_prediction_filepath(prediction_filepath, filepath):
    filename = filepath.split("/")[-1].split(".")[0]
    return os.path.join(prediction_filepath, filename + ".png")

def add_segm_predictions(dataset, normalize=True):

    for sample in dataset.iter_samples(progress=True, batch_size=1):
        fp = sample['segm_filepath']
        pred = cv2.imread(fp)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        if normalize:
            pred = pred / 255

        sample["predictions"] = fo.Segmentation(mask=pred.astype(int))
        sample.save()

def main(args):

    #data_path = cropped_test_images_path
    #labels_path = cropped_test_renamed_masks_path

    data_path = test_1_data_images_path
    labels_path = test_1_data_renamed_masks_path

    # Create the dataset
    dataset = fo.Dataset.from_dir(
        data_path=data_path,
        labels_path=labels_path,
        dataset_type=fo.types.ImageSegmentationDirectory,
        #mask_path = args.prediction_filepaths
    )

    filepaths = dataset.values("filepath")
    prediction_filepaths = [get_prediction_filepath(args.prediction_filepaths, fp) for fp in filepaths]

    dataset.set_values(
        "segm_filepath",
        prediction_filepaths
    )

    add_segm_predictions(dataset)

    ''' 
    # Evaluate the masks w/ ResNet50 backbone, treating the masks w/ ResNet101
    # backbone as "ground truth"
    results = dataset.evaluate_segmentations(
        "resnet50",
        gt_field="resnet101",
        eval_key="eval_simple",
    )

    # Get a sense for the per-sample variation in likeness
    print("Accuracy range: (%f, %f)" % dataset.bounds("eval_simple_accuracy"))
    print("Precision range: (%f, %f)" % dataset.bounds("eval_simple_precision"))
    print("Recall range: (%f, %f)" % dataset.bounds("eval_simple_recall"))
    '''

    #results = dataset.evaluate_segmentations(
    #    "predictions",
    #    gt_field="ground_truth",
    #    eval_key="eval_simple",
    #)

    # Print a classification report
    #results.print_report()

    session = fo.launch_app(dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--prediction_filepaths", default='./model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_1/segformer_processor_decoder_w_1_3_2_2024_03_22_09_36_51/test1/merged_model_results/', help="Path to the input image") #yolov8s-p2, rtdetr-l

    args = parser.parse_args()
    main(args)
