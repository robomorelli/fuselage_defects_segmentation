from scipy.ndimage import label, find_objects
from math import hypot
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from skimage.morphology import erosion
import torch
from skimage.morphology import remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize, binary_erosion, remove_small_holes

def F1Score(metrics, loss = 0):
    # compute performance measure for the current quantile filter
    tot_tp_test = metrics["TP"].sum()
    tot_fp_test = metrics["FP"].sum()
    tot_fn_test = metrics["FN"].sum()

    accuracy = (tot_tp_test + 0.001)/(tot_tp_test +
                                      tot_fp_test + tot_fn_test + 0.001)
    precision = (tot_tp_test + 0.001)/(tot_tp_test + tot_fp_test + 0.001)
    recall = (tot_tp_test + 0.001)/(tot_tp_test + tot_fn_test + 0.001)
    F1_score = 2*precision*recall/(precision + recall)

    return(F1_score, tot_tp_test, tot_fp_test, tot_fn_test, accuracy, precision, recall, loss)

def compute_metrics(mask, pred, metrics, img_name, obj_size=0):
    # extract predicted objects and counts
    pred_label, pred_count = ndimage.label(pred)
    #pred_label = remove_small_objects(pred_label, area_threshold=objs_size, connectivity=1)
    pred_objs = ndimage.find_objects(pred_label)


    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                             (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # extract target objects and counts
    targ_label, targ_count = ndimage.label(mask)
    targ_objs = ndimage.find_objects(targ_label)

    # compute centers of target objects
    targ_center = []
    for ob in targ_objs:
        targ_center.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                            (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # associate matching objects, true positives
    tp = 0
    fp = 0
    for pred_idx, pred_obj in enumerate(pred_objs):

        min_dist = 50  # 1.5-cells distance is the maximum accepted
        TP_flag = 0

        for targ_idx, targ_obj in enumerate(targ_objs):

            dist = hypot(pred_centers[pred_idx][0]-targ_center[targ_idx][0],
                         pred_centers[pred_idx][1]-targ_center[targ_idx][1])

            if dist < min_dist:

                TP_flag = 1
                min_dist = dist
                index = targ_idx

        if TP_flag == 1:
            tp += 1

            targ_center.pop(index)
            targ_objs.pop(index)

    # derive false negatives and false positives
    fn = targ_count - tp
    fp = pred_count - tp

    # update metrics dataframe
    print([tp, fp, fn])
    metrics.loc[img_name] = [tp, fp, fn, targ_count]

    return(metrics)


def compute_metrics_th(mask, pred, metrics, img_name, th=0.5, obj_size=0):
    # extract predicted objects and counts
    #pred = pred / 255.
    pred = (pred > th).astype(np.uint8) * 255

    pred = remove_small_objects(pred, min_size=obj_size, connectivity=1)

    pred_label, pred_count = ndimage.label(pred)
    pred_objs = ndimage.find_objects(pred_label)

    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                             (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # extract target objects and counts
    targ_label, targ_count = ndimage.label(mask)
    targ_objs = ndimage.find_objects(targ_label)

    # compute centers of target objects
    targ_center = []
    for ob in targ_objs:
        targ_center.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                            (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # associate matching objects, true positives
    tp = 0
    #targ_objs_origin = targ_objs.copy()
    for pred_index, ob_p in enumerate(pred_objs):
        for tar_index, ob_t in enumerate(targ_objs):

            intersection_x_min = max(ob_p[0].start, ob_t[0].start)
            intersection_x_max = min(ob_p[0].stop, ob_t[0].stop)  # the same object can't have the max less the min
            intersection_y_min = max(ob_p[1].start, ob_t[1].start)
            intersection_y_max = min(ob_p[1].stop, ob_t[1].stop)

            if (intersection_x_min < intersection_x_max) and (intersection_y_min < intersection_y_max):
                tp += 1
                targ_objs.pop(tar_index)
                break

    '''
    tp = 0
    fp = 0
    for pred_idx, pred_obj in enumerate(pred_objs):


        #min_dist = 50  # 1.5-cells distance is the maximum accepted
        #TP_flag = 0

        for targ_idx, targ_obj in enumerate(targ_objs):
             
            dist = hypot(pred_centers[pred_idx][0]-targ_center[targ_idx][0],
                         pred_centers[pred_idx][1]-targ_center[targ_idx][1])

            if dist < min_dist:

                TP_flag = 1
                min_dist = dist
                index = targ_idx
            
        if TP_flag == 1:
            tp += 1

            targ_center.pop(index)
            targ_objs.pop(index)
            
    # derive false negatives and false positives
    fn = targ_count - tp
    fp = pred_count - tp
    '''
    fn = targ_count - tp
    fp = pred_count - tp

    # update metrics dataframe
    #print([tp, fp, fn])
    metrics.loc[img_name] = [tp, fp, fn, targ_count]

    return(metrics)
