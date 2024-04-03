import os.path
import shutil
import argparse
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import sys
sys.path.append('..')
from dataset.segmentation import KFoldDataframeMulticlassProcessor_v2
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import *
from evaluation.utils import F1Score, compute_metrics_th, compute_metrics_multiclass, compute_iou_multiclass
from tqdm import tqdm
import cv2
import yaml
import torch.nn as nn
from transformers import (SegformerForSemanticSegmentation, SegformerImageProcessor)


AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(data_path, model_path, ths_num=0, normalize_imagenet=0
         , df_path=k_fold_data_path, split='test', save_into_common_folder=False
         ,save_into_model_folder=False, multi_ths=0, reduce_labels=1, ignore_index=255
         , f1_metrics=0, iou_metrics=1):

    if not os.path.exists(model_path):
        print('the model path is not correct')
        raise Exception

    save_path = os.path.join(Path(model_path).parent.as_posix())
    fold = os.path.basename(Path(model_path).parent.parent).split('_')[1]

    if 'train' in split:
        if "tot_bkg" in data_path:
            metrics_split = 'tot_bkg_metrics_train'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'tot_bkg_train'
        else:
            metrics_split = 'metrics_train'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'train'

    elif 'val' in split:
        if "tot_bkg" in data_path:
            metrics_split = 'tot_bkg_metrics_val'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'tot_bkg_val'
        else:
            metrics_split = 'metrics_val'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'val'

    elif 'test' in split:
        if "tot_bkg" in data_path:
            metrics_split = 'tot_bkg_metrics_test'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'tot_bkg_test'
        else:
            metrics_split = 'metrics_test'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'test'

    if not os.path.exists(os.path.join(save_path, metrics_split)):
        os.makedirs(metrics_path)
    else:
        shutil.rmtree(metrics_path)
        os.makedirs(metrics_path)

    num_classes = len(os.listdir(full_size_masks_classes_path))
    print('multiclass for n classes', num_classes)

    if 'model.' in model_path:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
    else:
        checkpoint = torch.load(os.path.join(model_path, "model.pth"), map_location=torch.device(device))

    cfg = checkpoint['cfg']

    if 'deeplab' in model_path:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, num_classes=num_classes + 1)
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'segformer' in model_path:
        encoder_name = cfg.model.encoder_name
        print(f' segformer with {encoder_name} encoder')

        with open('../preprocessing/class_mapping.yaml', 'r') as stream:
            try:
                # Converts yaml document to python object
                label2id = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
        label2id['bkg'] = 0
        id2label = {v: k for k, v in label2id.items()}
        model = SegformerForSemanticSegmentation.from_pretrained(encoder_name,
                                                                 num_labels=num_classes + 1,
                                                                 id2label=id2label,
                                                                 label2id=label2id,
                                                                 ignore_mismatched_sizes=True)
        print('classes dict', id2label)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)

    if 'cfg' in checkpoint.keys():
        cfg = checkpoint['cfg']
        normalize_imagenet = cfg.dataset.normalize_imagenet
        print('from cfg infer if normalize imagenet....that is', normalize_imagenet)
    else:
        normalize_imagenet = 0
        print('not cfg', normalize_imagenet)

    if not cfg.opt.processor:
        processor = None
    else:
        processor = SegformerImageProcessor.from_pretrained(cfg.model.encoder_name)

    transform = None
    df_path = os.path.join(df_path, f"fold_{fold}", split)

    dataset = KFoldDataframeMulticlassProcessor_v2(data_path, df_path=df_path, transform=transform,
                                                normalize_imagenet=normalize_imagenet, cropped=False,
                                                from_full_to_crop=True,  processor=processor)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    if ths_num > 0 and multi_ths:
        ths = np.linspace(0.2, 0.95, ths_num)
    else:
        ths = [0.5]

    if f1_metrics:
        metrics_dicts = {f"{th}": pd.DataFrame(None, columns=["TP", "FP", "FN", "target objects"]) for th in ths}
        global_metrics = pd.DataFrame(None, columns=["F1", "TP", "FP", "FN", "accuracy", "precision", "recall", "loss"])
    elif iou_metrics:
        columns = [f'class_{ix+1}_iou' for ix in range(num_classes)]
        columns = columns + [f'class_{ix+1}_accuracy' for ix in range(num_classes)]
        columns = columns + ["loss"]
        global_metrics = pd.DataFrame(None, columns=columns)

        results_dict_iou = {f'class_{ix + 1}_iou': [] for ix in range(num_classes)}
        results_dict_accuracy = ({f'class_{ix + 1}_accuracy': [] for ix in range(num_classes)})

        results_dict_summary_iou = {f'class_{ix + 1}_iou_summary': [] for ix in range(num_classes)}
        results_dict_summary_accuracy = {f'class_{ix + 1}_accuracy_summary': [] for ix in range(num_classes)}

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    filenames = dataset.images_file_names

    if save_into_common_folder:
        for th in ths:
            if split == 'train':
                save_into_common_path = os.path.join(common_path_train_results, f"model_results_{th}")
                save_into_common_path_viz = os.path.join(common_path_train_results_viz, f"model_results_{th}")
            elif split == 'val':
                save_into_common_path = os.path.join(common_path_val_results, f"model_results_{th}")
                save_into_common_path_viz = os.path.join(common_path_val_results_viz, f"model_results_{th}")
            elif split == 'test':
                save_into_common_path = os.path.join(common_path_test_results, f"model_results_{th}")
                save_into_common_path_viz = os.path.join(common_path_test_results_viz, f"model_results_{th}")
            else:
                raise NotImplementedError
            if not os.path.exists(save_into_common_path):
                os.makedirs(save_into_common_path)
            else:
                shutil.rmtree(save_into_common_path)
                os.makedirs(save_into_common_path)
            if not os.path.exists(save_into_common_path_viz):
                os.makedirs(save_into_common_path_viz)
            else:
                shutil.rmtree(save_into_common_path_viz)
                os.makedirs(save_into_common_path_viz)


    if save_into_model_folder:
        for th in ths:
            if split == 'train':
                save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
                save_into_model_path_viz = os.path.join(save_path, f"{split}/model_results_{th}_viz")
            elif split == 'val':
                save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
                save_into_model_path_viz = os.path.join(save_path, f"{split}/model_results_{th}_viz")
            elif split == 'test':
                save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
                save_into_model_path_viz = os.path.join(save_path, f"{split}/model_results_{th}_viz")
            else:
                raise NotImplementedError
            if not os.path.exists(save_into_model_path):
                os.makedirs(save_into_model_path)
            else:
                shutil.rmtree(save_into_model_path)
                os.makedirs(save_into_model_path)
            if not os.path.exists(save_into_model_path_viz):
                os.makedirs(save_into_model_path_viz)
            else:
                shutil.rmtree(save_into_model_path_viz)
                os.makedirs(save_into_model_path_viz)



    with torch.no_grad():
        running_loss = 0.0
        for i, (im, gt_mask) in tqdm(enumerate(dataloader), total=len(dataset)):

            if gt_mask.max() > num_classes:
                for i in range(num_classes):
                    gt_mask[gt_mask == np.ceil(255 / (i + 1))] = num_classes - i

            name = filenames[i]
            logits = model(im.to(device)).logits

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=tuple(im.shape[-2:]),  # (height, width)
                mode='bilinear',
                align_corners=False
            )

            loss = criterion(upsampled_logits.float(), gt_mask.to(device).squeeze(1).long()).item()

            print('mask', np.unique(gt_mask))
            pred_mask = upsampled_logits[0].softmax(0).permute(1, 2, 0).detach().cpu().numpy()
            pred_mask = np.argmax(pred_mask, 2)
            print('pred', np.unique(pred_mask))

            gt_fh = dataset.images_file_names[i]
            gt_mask = gt_mask.detach().cpu().numpy()
            running_loss += loss
            mean_loss = running_loss / (i + 1)
            print('loss', mean_loss)

            if not multi_ths:
                results, pred = compute_iou_multiclass(gt_mask, pred_mask, img_name=gt_fh, obj_size=args.remove_small_objs_size
                                                 , reduce_labels=reduce_labels, ignore_index=ignore_index)

                for i, k in enumerate(list(results_dict_iou.keys())):
                    results_dict_iou[f'class_{i+1}_iou'].append(results['per_category_iou'][i])
                for i, k in enumerate(list(results_dict_accuracy.keys())):
                    results_dict_accuracy[f'class_{i+1}_accuracy'].append(results['per_category_iou'][i])


            else:

                for th in ths:
                    metrics = metrics_dicts[str(th)]
                    metrics_dicts[th], pred = compute_metrics_th(gt_mask, pred_mask,
                                                           img_name=gt_fh, metrics=metrics,
                                                           th=th, obj_size=args.remove_small_objs_size, return_pred_th=True)

            if save_into_common_folder:
                if split == 'train':
                    save_into_common_path = os.path.join(common_path_train_results, f"model_results_{th}")
                    save_into_common_path_viz = os.path.join(common_path_train_results_viz,
                                                             f"model_results_{th}")
                elif split == 'val':
                    save_into_common_path = os.path.join(common_path_val_results, f"model_results_{th}")
                    save_into_common_path_viz = os.path.join(common_path_val_results_viz, f"model_results_{th}")
                elif split == 'test':
                    save_into_common_path = os.path.join(common_path_test_results, f"model_results_{th}")
                    save_into_common_path_viz = os.path.join(common_path_test_results_viz,
                                                             f"model_results_{th}")
                cv2.imwrite(os.path.join(save_into_common_path, f"{name}"), np.squeeze(pred))
                pred = (np.array(pred) / num_classes) * 255
                cv2.imwrite(os.path.join(save_into_common_path_viz, f"{name}"), np.squeeze(pred))

            if save_into_model_folder:
                cv2.imwrite(os.path.join(save_into_model_path, f"{name}"), np.squeeze(pred))
                pred =  (np.array(pred) / num_classes) * 255
                cv2.imwrite(os.path.join(save_into_model_path_viz, f"{name}"), np.squeeze(pred))

            if multi_ths:
                if f1_metrics:
                    for th in ths:
                        metrics = metrics_dicts[str(th)]
                        outname = os.path.join(metrics_path, f'{split_suffix }_metrics_{th}.csv')
                        metrics.to_csv(outname, index=True)
                        global_metrics.loc[th] = F1Score(metrics, mean_loss)  # possible to itera on different threshold


    if multi_ths:
        outname = os.path.join(save_path, f'{split_suffix}_global_metrics.csv')
        global_metrics.to_csv(outname, index=True, index_label='Threshold')

    else:
        if not f1_metrics:
            global_values = []
            for k in list(results_dict_iou.keys()):
                results_dict_iou[k] = [x for x in results_dict_iou[k] if not (np.isnan(x) or x == 0)]
                results_dict_summary_iou[k] = np.mean(results_dict_iou[k])
                global_values.append(results_dict_summary_iou[k])
            for k in list(results_dict_accuracy.keys()):
                results_dict_accuracy[k] = [x for x in results_dict_accuracy[k] if not (np.isnan(x) or x == 0)]
                results_dict_summary_accuracy[k] = np.mean(results_dict_accuracy[k])
                global_values.append(results_dict_summary_accuracy[k])

            global_metrics.iloc[0,:] = global_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--ths_num", default=7, help="how many ths from 0.2 to 0.95")
    parser.add_argument("--normalize_imagenet", default=0, help="imagenet normalization")
    parser.add_argument("--model_path",
                        default="../model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_1/segformer_processor_decoder_w_1_3_2_2024_03_22_09_36_51/model.pth"
                        , help="Path to the input model")
    parser.add_argument("--data_path", default=cropped_tot_bkg_data_path
                        , help="Path to the input model")
    parser.add_argument("--df_path", default=k_fold_data_path
                        , help="Path to the input model")
    #parser.add_argument("--fold", default=7
    #                    , help="Path to the input model")
    parser.add_argument("--split", default="test"
                        , help="Path to the input model")
    parser.add_argument("--remove_small_objs_size", default=100, help="")
    parser.add_argument("--save_into_common_folder", default=0
                        , help="Path to the input model")
    parser.add_argument("--save_into_model_folder", default=1
                        , help="Path to the input model")
    parser.add_argument("--multi_ths", default=0, help="")
    parser.add_argument("--reduce_labels", default=0
                        , help="Path to the input model")
    parser.add_argument("--ignore_index", default=255, help="")
    parser.add_argument("--f1_metrics", default=0, help="")
    parser.add_argument("--iou_metrics", default=1, help="")

    args = parser.parse_args()
    main(data_path=args.data_path, model_path=args.model_path, ths_num=args.ths_num
         , df_path=args.df_path, split=args.split, save_into_common_folder=args.save_into_common_folder,
         save_into_model_folder = args.save_into_model_folder, multi_ths=args.multi_ths,
         reduce_labels=args.reduce_labels, ignore_index=args.ignore_index
         ,f1_metrics=args.f1_metrics, iou_metrics=args.iou_metrics)


