import shutil
import argparse
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import sys
sys.path.append('..')
from dataset.segmentation import BinarySegmentationPil, BinarySegmentationAlb
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import *
from evaluation.utils import compute_metrics, F1Score, compute_metrics_th

import matplotlib
AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(images_path, model_path, ths_num=0, normalize_imagenet=0):

    save_path = Path(model_path).parent.as_posix()
    if 'train' in images_path:
        if "tot_bkg" in images_path:
            metrics_split = 'tot_bkg_metrics_train'
            metrics_path = os.path.join(save_path, metrics_split)
            split = 'tot_bkg_train'
        else:
            metrics_split = 'metrics_train'
            metrics_path = os.path.join(save_path, metrics_split)
            split = 'train'
    elif 'val' in images_path:
        if "tot_bkg" in images_path:
            metrics_split = 'tot_bkg_metrics_val'
            metrics_path = os.path.join(save_path, metrics_split)
            split = 'tot_bkg_val'
        else:
            metrics_split = 'metrics_val'
            metrics_path = os.path.join(save_path, metrics_split)
            split = 'val'

    elif 'test1' in images_path:
        if "tot_bkg" in images_path:
            metrics_split = 'tot_bkg_metrics_test'
            metrics_path = os.path.join(save_path, metrics_split)
            split = 'tot_bkg_test'
        else:
            metrics_split = 'metrics_test'
            metrics_path = os.path.join(save_path, metrics_split)
            split = 'test1'

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

    data_path = Path(images_path).parent.as_posix()

    dataset = BinarySegmentationAlb(data_path, transform=None, test=False, normalize_imagenet=normalize_imagenet)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    global_metrics = pd.DataFrame(None, columns=["F1", "TP", "FP", "FN", "accuracy", "precision", "recall"])

    if ths_num > 0:
        ths = np.linspace(0.2, 0.95, ths_num)
    else:
        ths = [0.5]

    metrics_dicts = {f"{th}":pd.DataFrame(None, columns=["TP", "FP", "FN", "target objects"]) for th in ths}

    model.eval()

    with torch.no_grad():
        for i, (im, gt_mask) in enumerate(dataloader):
            if 'deeplab' or 'resnet' in model_path:
                pred_mask = model(im.to(device))["out"].sigmoid().detach().cpu().numpy()
            else:
                pred_mask = model(im.to(device)).detach().cpu().numpy()
            gt_mask = gt_mask.detach().cpu().numpy()
            gt_fh = dataset.images_file_names[i]

            for th in ths:
                metrics = metrics_dicts[str(th)]
                metrics_dicts[th] = compute_metrics_th(gt_mask, pred_mask,
                                                       img_name=gt_fh, metrics=metrics,
                                                       th=th, obj_size=args.remove_small_objs_size)

        for th in ths:
            metrics = metrics_dicts[str(th)]
            outname = os.path.join(metrics_path, f'{split}_metrics_{th}.csv')
            metrics.to_csv(outname, index=True)
            global_metrics.loc[th] = F1Score(metrics)  #possible to itera on different threshold

    outname = os.path.join(save_path, f'{split}_global_metrics.csv')
    global_metrics.to_csv(outname, index=True, index_label='Threshold')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--images_path", default=cropped_test_images_path, help="Path to the input image")
    parser.add_argument("--ths_num", default=7, help="how many ths from 0.2 to 0.95")
    parser.add_argument("--normalize_imagenet", default=0, help="imagenet normalization")
    parser.add_argument("--model_path", default="../model_results/deeplab/deeplabv3_resnet101/deeplab_bkg_025_2024_02_11_19_27_26/model.pth"
                        , help="Path to the input model")
    parser.add_argument("--remove_small_objs_size", default=150, help="")

    args = parser.parse_args()
    main(images_path=args.images_path, model_path=args.model_path, ths_num=args.ths_num,
         normalize_imagenet=args.normalize_imagenet)
