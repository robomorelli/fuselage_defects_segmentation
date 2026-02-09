import os.path
import shutil
import argparse
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import sys
sys.path.append('..')
from dataset.segmentation import KFoldDataframeMulticlassProcessor
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import *
from evaluation.eval_utils import F1Score, compute_metrics_th, compute_metrics_multiclass, compute_iou_multiclass
from tqdm import tqdm
import cv2
import yaml
import torch.nn as nn
from transformers import (SegformerForSemanticSegmentation, SegformerImageProcessor)


AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(data_path, model_path, exp_name=None):

    num_classes = len(os.listdir(full_size_masks_classes_path))
    print('multiclass for n classes', num_classes)

    if not os.path.exists(model_path):
        print('the model path is not correct')
        raise Exception

    save_path = os.path.join(Path(model_path).parent.as_posix())
    if exp_name != None:
        save_path = os.path.join(save_path, exp_name)
    else:
        save_path = os.path.join(save_path, os.path.basename(Path(data_path).parent))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

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
    else:
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
    dataset = KFoldDataframeMulticlassProcessor(data_path, df_path=None, from_folder=True, transform=transform,
                                                normalize_imagenet=normalize_imagenet,
                                                from_full_to_crop=True,  processor=processor)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    filenames = dataset.images_file_names

    save_into_model_path = os.path.join(save_path, f"model_results")
    save_into_model_path_viz = os.path.join(save_path, f"model_results_viz")

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

            #loss = criterion(upsampled_logits.float(), gt_mask.to(device).squeeze(1).long()).item()

            pred_mask = upsampled_logits[0].softmax(0).permute(1, 2, 0).detach().cpu().numpy()
            pred_mask = np.argmax(pred_mask, 2)
            print('pred', np.unique(pred_mask))

            cv2.imwrite(os.path.join(save_into_model_path, f"{name}"), np.squeeze(pred_mask))
            pred_mask =  (np.array(pred_mask) / num_classes) * 255
            cv2.imwrite(os.path.join(save_into_model_path_viz, f"{name}"), np.squeeze(pred_mask))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--model_path",
                        default="../model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_4/segformer_processor_decoder_w_1_3_2_2024_03_27_14_31_29/model.pth"
                        , help="Path to the input model")
    parser.add_argument("--data_path", default=cropped_may_data_path
                        , help="Path to the input model")
    #parser.add_argument("--remove_small_objs_size", default=100, help="")
    parser.add_argument("--exp_name", default=None, help="Path to the input model")
    #parser.add_argument("--draw_contour", default=100, help="")

    args = parser.parse_args()
    main(data_path=args.data_path, model_path=args.model_path, exp_name=args.exp_name)


