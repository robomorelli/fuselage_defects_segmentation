import os.path
import shutil
import argparse
import numpy as np
import sys
sys.path.append('..')
import torch
from pathlib import Path
from config import *
import cv2

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    colors = [(0, 255, 0),  # Green
              (0, 0, 255),  # Red
              (255, 0, 0),  # Blue
              (255, 255, 0),  # Cyan
              (255, 0, 255),  # Magenta
              (0, 255, 255),  # Yellow
              (128, 0, 0),  # Maroon
              (0, 128, 0)]  # Olive


    num_classes = len(os.listdir(full_size_masks_classes_path))
    print('multiclass for n classes', num_classes)

    predictions_path = args.predictions_folder
    images_path = args.images_folder

    if not os.path.exists(predictions_path):
        print('the model path is not correct')
        raise Exception

    save_path = os.path.join(Path(predictions_path).parent.as_posix())
    save_path = os.path.join(save_path, 'filtered_objects')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    pred_names = os.listdir(predictions_path)
    preds_fhs = [os.path.join(predictions_path, x) for x in pred_names]

    id_classes = [i + 1 for i in range(num_classes)]

    kernel = np.ones((15, 15), np.uint8)  # Adjust the kernel size as needed

    for ix, pred_fh in enumerate(preds_fhs):
        name = os.path.basename(pred_fh)
        bboxes_dict = {str(k): [] for k in id_classes}
        pred = cv2.imread(pred_fh)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        if 'mask' in name:
            name.replace('_mask.', '.')
        image = cv2.imread(os.path.join(images_path, f"{name}"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for class_value in id_classes:

            #other_classes = [x for x in id_classes if x != class_value]
            # Create a mask for the current class value
            class_mask = np.uint8(pred == class_value)
            #class_mask_d = cv2.dilate(class_mask, kernel, iterations=0)
            #contours_d, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for ix, contour in enumerate(contours):
                remove = False
                # Get the bounding rectangle of each contour
                #x0, y0, w0, h0 = cv2.boundingRect(contours[ix])

                x1, y1, w1, h1 = cv2.boundingRect(contour)
                x1_right = x1 + w1
                y1_bottom = y1 + h1

                for other_class_value in id_classes:
                    other_class_mask = np.uint8(pred == other_class_value)
                    other_contours, _ = cv2.findContours(np.squeeze(other_class_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(other_contours) > 0:
                        for iox, o_contour in enumerate(other_contours):

                            x2, y2, w2, h2 = cv2.boundingRect(o_contour)
                            if [x2, y2, w2, h2] == [x1, y1, w1, h1]:
                                continue

                            x2_right = x2 + w2
                            y2_bottom = y2 + h2

                            # Calculate the intersection coordinates
                            intersection_left = max(x1, x2)
                            intersection_top = max(y1, y2)
                            intersection_right = min(x1_right, x2_right)
                            intersection_bottom = min(y1_bottom, y2_bottom)

                            # Calculate the width and height of the intersection
                            intersection_width = max(0, intersection_right - intersection_left)
                            intersection_height = max(0, intersection_bottom - intersection_top)

                            # Calculate the area of intersection
                            intersection_area = intersection_width * intersection_height

                            left_dist = x2 - (x1 + w1)
                            right_dist = x1 - (x2 + w2)
                            top_dist = y2 - (y1 + h1)
                            bottom_dist = y1 - (y2 + h2)
                            threshold = 0

                            if (abs(left_dist) <= threshold or abs(right_dist) <=
                                    threshold or abs(top_dist) <= threshold or abs(bottom_dist) <= threshold) or intersection_area > 0:
                                if w2*h2 > w1*h1:
                                    #class_name = other_class_value
                                    remove = True
                                    break

                                #cont = np.vstack(([x1, y1, w1, h1], [x2, y2, w2, h2]))
                                #x2, y2, w2, h2 = cv2.boundingRect(cont)
                            ''' 
                            elif intersection_area > 0:
                                
                                if w2 * h2 > w1 * h1:
                                    class_name = other_class_value
                                    remove = True
   
                                    removed_mask = np.zeros_like(pred)
                                     
                                    # Draw the contour on the new mask with color white (255) and thickness -1 (filled)
                                    cv2.drawContours(removed_mask, [contours[ix]], 0, (255), -1)
                                    # Invert the mask (convert white to black and vice versa)
                                    removed_mask = cv2.bitwise_not(removed_mask)
                                    # Perform bitwise AND operation to remove the contour from the original mask
                                    pred = cv2.bitwise_and(pred, pred, mask=removed_mask)

                                    removed_mask = np.zeros_like(pred)
                                    cv2.drawContours(removed_mask, [other_contours[iox]], 0, (255), -1)
                                    # Invert the mask (convert white to black and vice versa)
                                    removed_mask = cv2.bitwise_not(removed_mask)
                                    pred = cv2.bitwise_and(pred, pred, mask=removed_mask)

                                    merged_contour = np.vstack(([x1, y1, w1, h1], other_contours[iox]))
                                    x0, y0, w0, h0 = cv2.boundingRect(merged_contour)
                                    #merged_image = np.zeros_like(pred)
                                    cv2.drawContours(pred, [merged_contour], -1, (0, 255, 0), 2)
                                    
                                else:
                                    bboxes_dict[str(class_value)].append([x0, y0, w0, h0])
                                    removed_mask = np.zeros_like(pred)
                                    # Draw the contour on the new mask with color white (255) and thickness -1 (filled)
                                    cv2.drawContours(removed_mask, [other_contours[iox]], 0, (255), -1)
                                    # Invert the mask (convert white to black and vice versa)
                                    removed_mask = cv2.bitwise_not(removed_mask)
                                    # Perform bitwise AND operation to remove the contour from the original mask
                                    pred = cv2.bitwise_and(pred, pred, mask=removed_mask)
                                '''

                    if not remove:
                        bboxes_dict[str(class_value)].append([x1, y1, w1, h1])
                    else:
                        bboxes_dict[str(other_class_value)].append([x2, y2, w2, h2])

        for k, boxes in bboxes_dict.items():
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), colors[int(k)], 2)

        cv2.imwrite(os.path.join(save_path, f"{name}"), np.squeeze(image))
        #pred_mask = (np.array(pred_mask) / num_classes) * 255
        #cv2.imwrite(os.path.join(save_into_model_path_viz, f"{name}"), np.squeeze(pred_mask))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--predictions_folder",
                        default="../model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_9_hpc/test_1/merged_model_results/")
    parser.add_argument("--images_folder", default=test_1_data_images_path, help="full size image path")

    args = parser.parse_args()
    main(args)


