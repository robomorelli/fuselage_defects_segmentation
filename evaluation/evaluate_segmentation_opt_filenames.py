import os.path
import shutil
import argparse
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import sys
sys.path.append('..')
from dataset.segmentation import KFoldDataframe, BinarySegmentationAlb
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import *
from evaluation.utils import compute_metrics, F1Score, compute_metrics_th
from tqdm import tqdm
import cv2


AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(data_path, model_path, ths_num=0, normalize_imagenet=0
         , df_path=k_fold_data_path, fold=1, split='test', save_into_common_folder=False
         ,save_into_model_folder=False):

    save_path = os.path.join(Path(model_path).parent.parent.parent.as_posix(), f'fold_{fold}'
                             , os.path.basename(Path(model_path).parent.as_posix()))

    if 'train' in split:
        if "tot_bkg" in data_path:
            metrics_split = 'tot_bkg_metrics_train'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix  = 'tot_bkg_train'
        else:
            metrics_split = 'metrics_train'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix  = 'train'

    elif 'val' in split:
        if "tot_bkg" in data_path:
            metrics_split = 'tot_bkg_metrics_val'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix = 'tot_bkg_val'
        else:
            metrics_split = 'metrics_val'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix  = 'val'

    elif 'test' in split:
        if "tot_bkg" in data_path:
            metrics_split = 'tot_bkg_metrics_test'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix  = 'tot_bkg_test'
        else:
            metrics_split = 'metrics_test'
            metrics_path = os.path.join(save_path, metrics_split)
            split_suffix  = 'test'

    if not os.path.exists(os.path.join(save_path, metrics_split)):
        os.makedirs(metrics_path)
    else:
        shutil.rmtree(metrics_path)
        os.makedirs(metrics_path)

    if 'deeplab' in model_path:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, num_classes=1)
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if 'cfg' in checkpoint.keys():
        cfg = checkpoint['cfg']

    if 'cfg' in checkpoint.keys():
        normalize_imagenet = cfg.dataset.normalize_imagenet
    else:
        normalize_imagenet = normalize_imagenet

    df_path = os.path.join(df_path, f"fold_{fold}", split)

    dataset = KFoldDataframe(data_path, df_path=df_path, transform=None, test=False,
                             normalize_imagenet=normalize_imagenet, cropped=False, from_full_to_crop=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    global_metrics = pd.DataFrame(None, columns=["F1", "TP", "FP", "FN", "accuracy", "precision", "recall", "loss"])

    if ths_num > 0:
        ths = np.linspace(0.2, 0.95, ths_num)
    else:
        ths = [0.5]

    metrics_dicts = {f"{th}": pd.DataFrame(None, columns=["TP", "FP", "FN", "target objects"]) for th in ths}

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    filenames = dataset.images_file_names

    if save_into_common_folder:
        for th in ths:
            if split == 'train':
                save_into_common_path = os.path.join(common_path_train_results, f"model_results_{th}")
            elif split == 'val':
                save_into_common_path = os.path.join(common_path_val_results, f"model_results_{th}")
            elif split == 'test':
                save_into_common_path = os.path.join(common_path_test_results, f"model_results_{th}")
            else:
                raise NotImplementedError
            if not os.path.exists(save_into_common_path):
                os.makedirs(save_into_common_path)
            else:
                shutil.rmtree(save_into_common_path)
                os.makedirs(save_into_common_path)

    if save_into_model_folder:
        for th in ths:
            if split == 'train':
                save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
            elif split == 'val':
                save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
            elif split == 'test':
                save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
            else:
                raise NotImplementedError
            if not os.path.exists(save_into_model_path):
                os.makedirs(save_into_model_path)
            else:
                shutil.rmtree(save_into_model_path)
                os.makedirs(save_into_model_path)

    with torch.no_grad():
        running_loss = 0.0
        for i, (im, gt_mask) in tqdm(enumerate(dataloader), total=len(dataset)):

            name = filenames[i]
            if 'deeplab' or 'resnet' in model_path:
                pred_mask = model(im.to(device))["out"]
            else:
                pred_mask = model(im.to(device))

            gt_fh = dataset.images_file_names[i]

            loss = criterion(pred_mask, gt_mask.to(device))
            pred_mask = pred_mask.sigmoid().detach().cpu().numpy()
            gt_mask = gt_mask.detach().cpu().numpy()
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
            print(running_loss / (i+1))

            for th in ths:

                metrics = metrics_dicts[str(th)]
                metrics_dicts[th], pred = compute_metrics_th(gt_mask, pred_mask,
                                                       img_name=gt_fh, metrics=metrics,
                                                       th=th, obj_size=args.remove_small_objs_size, return_pred_th=True)

                if save_into_common_folder:
                    if split == 'train':
                        save_into_common_path = os.path.join(common_path_train_results, f"model_results_{th}")
                    elif split == 'val':
                        save_into_common_path = os.path.join(common_path_val_results, f"model_results_{th}")
                    elif split == 'test':
                        save_into_common_path = os.path.join(common_path_test_results, f"model_results_{th}")
                    cv2.imwrite(os.path.join(save_into_common_path, f"{name}"), np.squeeze(pred * 255))
                if save_into_model_folder:
                    if split == 'train':
                        save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
                    elif split == 'val':
                        save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
                    elif split == 'test':
                        save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
                    cv2.imwrite(os.path.join(save_into_model_path, f"{name}"), np.squeeze(pred * 255))


        for th in ths:
            metrics = metrics_dicts[str(th)]
            outname = os.path.join(metrics_path, f'{split_suffix }_metrics_{th}.csv')
            metrics.to_csv(outname, index=True)
            global_metrics.loc[th] = F1Score(metrics, mean_loss)  # possible to itera on different threshold

    outname = os.path.join(save_path, f'{split_suffix}_global_metrics.csv')
    global_metrics.to_csv(outname, index=True, index_label='Threshold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--ths_num", default=7, help="how many ths from 0.2 to 0.95")
    parser.add_argument("--normalize_imagenet", default=0, help="imagenet normalization")
    parser.add_argument("--model_path",
                        default="../model_results/deeplab_k_fold/deeplabv3_resnet101/fold_1/deeplab_k_fold_2024_02_16_15_35_45/model.pth"
                        , help="Path to the input model")
    parser.add_argument("--data_path", default=cropped_tot_bkg_data_path
                        , help="Path to the input model")
    parser.add_argument("--df_path", default=k_fold_data_path
                        , help="Path to the input model")
    parser.add_argument("--fold", default=2
                        , help="Path to the input model")
    parser.add_argument("--split", default="test"
                        , help="Path to the input model")
    parser.add_argument("--remove_small_objs_size", default=100, help="")
    parser.add_argument("--save_into_common_folder", default=1
                        , help="Path to the input model")
    parser.add_argument("--save_into_model_folder", default=0
                        , help="Path to the input model")

    args = parser.parse_args()
    main(data_path=args.data_path, model_path=args.model_path, ths_num=args.ths_num
         , df_path=args.df_path, fold=args.fold, split=args.split, save_into_common_folder=args.save_into_common_folder,
         save_into_model_folder = args.save_into_model_folder)


