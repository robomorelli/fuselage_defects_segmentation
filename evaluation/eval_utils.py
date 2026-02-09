from scipy.ndimage import label, find_objects
from math import hypot
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from skimage.morphology import erosion
import torch
from skimage.morphology import remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize, binary_erosion, remove_small_holes
import evaluate

mean_iou = evaluate.load("mean_iou")

def  F1Score_multiclass(metrics, mean_loss, class_names, global_columns):

    global_metrics_dict = {gc: 0 for gc in global_columns}

    for cl_name in class_names:

        global_metrics_dict[f"{cl_name}_TP"] = metrics[f"{cl_name}_TP"].sum()
        global_metrics_dict[f"{cl_name}_FP"] = metrics[f"{cl_name}_FP"].sum()
        global_metrics_dict[f"{cl_name}_FN"] = metrics[f"{cl_name}_FN"].sum()

        global_metrics_dict[f"{cl_name}_accuracy"] = (global_metrics_dict[f"{cl_name}_TP"] + 0.001) / (global_metrics_dict[f"{cl_name}_TP"] +
                                            global_metrics_dict[f"{cl_name}_FP"]+ global_metrics_dict[f"{cl_name}_FN"] + 0.001)
        global_metrics_dict[f"{cl_name}_precision"]  = ((global_metrics_dict[f"{cl_name}_TP"] + 0.001) /
                                                        (global_metrics_dict[f"{cl_name}_TP"]+ global_metrics_dict[f"{cl_name}_FP"] + 0.001))
        global_metrics_dict[f"{cl_name}_recall"]  = ((global_metrics_dict[f"{cl_name}_TP"] + 0.001) /
                                                     (global_metrics_dict[f"{cl_name}_TP"] + global_metrics_dict[f"{cl_name}_FN"] + 0.001))
        global_metrics_dict[f"{cl_name}_F1"] = (2 * global_metrics_dict[f"{cl_name}_precision"] *
                                                global_metrics_dict[f"{cl_name}_recall"]/ (global_metrics_dict[f"{cl_name}_precision"]
                                                                                           + global_metrics_dict[f"{cl_name}_recall"]))

    return list(global_metrics_dict.values())

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


def compute_metrics_th(mask, pred, metrics, img_name, th=0.5, obj_size=0, return_pred_th=True):
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

    if return_pred_th:
        return (metrics), pred
    else:
        return(metrics)



def compute_iou_multiclass(mask, pred, img_name,  n_classes=2, obj_size=0, ignore_index=255, reduce_labels=True):

    if mask.max() > n_classes:
        for i in range(n_classes):
            mask[mask==np.ceil(255/(i+1))]=n_classes-i

    pred_mask = remove_small_objects(pred, min_size=obj_size, connectivity=1).astype(np.int8)
    pred = (pred_mask - 1).astype(np.uint16)
    pred = np.clip(pred, 0, 255)
    pred = pred.astype(np.uint16)
    mask = np.squeeze(mask).astype(np.uint16)
    #results = mean_iou.compute(predictions=[pred], references=[mask], num_labels=n_classes,
    #                           ignore_index=255, reduce_labels=1)
    results= None

    return results, pred_mask


def compute_metrics_multiclass(mask, pred, metrics, img_name, th = 0.3,
                               n_classes=2, obj_size=0, return_pred_th=True, id_labels_dict=None):


    #print(np.unique(mask), np.unique(pred))
    #if (2 in np.unique(pred)) and (2 in np.unique(mask)):
    #    print('graffio')

    pred_original = pred.copy()

    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    # Conversion to multichannel prediction
    channels = [np.zeros_like(pred) for _ in range(n_classes)]
    mask_channels = [np.zeros_like(mask) for _ in range(n_classes)]

    # Assign 1 to each channel where tensor equals the channel index
    for i in range(n_classes):
        channels[i][pred == i + 1] = 1

    # Stack the channels to form a multi-channel tensor
    # Stack the channels to form a multi-channel tensor
    pred = np.stack(channels, axis=0)
    pred = np.squeeze(pred)

    # The same for the mask
    for i in range(n_classes):
        mask_channels[i][mask == i + 1] = 1

    # Stack the channels to form a multi-channel tensor
    # Stack the channels to form a multi-channel tensor
    mask = np.stack(mask_channels, axis=0)
    mask = np.squeeze(mask)

    # previous pre processing: converto to 255 - remove small objects - count object
    # apply channel by channel
    results_dict = {k: 0 for k in list(metrics.columns)}
    for id_ch in range(pred.shape[0]):
        ch = pred[id_ch]
        ch_mask = mask[id_ch]

        ch = (ch > th).astype(np.uint8)
        ch = remove_small_objects(ch.astype(bool), min_size=obj_size, connectivity=1).astype(np.uint8) * 255

        pred_label, pred_count = ndimage.label(ch)
        pred_objs = ndimage.find_objects(pred_label)

        # compute centers of predicted objects
        pred_centers = []
        for ob in pred_objs:
            pred_centers.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                                 (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

        # extract target objects and counts
        targ_label, targ_count = ndimage.label(ch_mask)
        targ_objs = ndimage.find_objects(targ_label)

        # compute centers of target objects
        targ_center = []
        for ob in targ_objs:
            targ_center.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                                (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

        # associate matching objects, true positives
        tp = 0
        # targ_objs_origin = targ_objs.copy()
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

        fn = targ_count - tp
        fp = pred_count - tp
        results_list = [tp, fp, fn]
        results_mapping = ['TP', 'FP', 'FN']
        # append the tp fp fn for the i-th channel
        results_dict.update({id_labels_dict[id_ch+1]+'_'+f'{results_mapping[ix]}': res for ix, res in enumerate(results_list)})

    results_to_df = [results_dict[k] for k in list(metrics.columns)]
    # update metrics dataframe
    # print([tp, fp, fn])
    metrics.loc[img_name] = results_to_df
    #metrics.append(results_dict, ignore_index=True)
    #print(metrics.loc[img_name])

    if return_pred_th:
        return (metrics), pred_original
    else:
        return (metrics)
