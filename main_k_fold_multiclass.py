import argparse
import os
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from dataset.segmentation import KFoldDataframe, BinarySegmentationPil, KFoldDataframeMulticlass, KFoldDataframeMulticlassProcessor_v2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (RandomCrop, CenterCrop, ElasticTransform, RGBShift, Rotate,
                            Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,
                            Transpose,
                            ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip,
                            HorizontalFlip,
                            HueSaturationValue,
                            )
from transformers import (
    SegformerForSemanticSegmentation,
    TrainingArguments, Trainer,
    SegformerImageProcessor)
import yaml
import torch
import torch.optim
import segmentation_models_pytorch as smp
from utils.training import training_cycle, training_cycle_deeplab, training_cycle_deeplab_multiclass, training_cycle_segformer_multiclass
from utils.opt import EarlyStopping
import yaml
import json
import types
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import datetime
from model.resunet import *
from config import *

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_object(dct):
    return types.SimpleNamespace(**dct)


def read_config(config_name):
    with open(os.path.join(config_name), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)
    return cfg


def main(args):
    current_date = datetime.datetime.now()

    now = current_date.strftime("%Y_%m_%d_%H_%M_%S")
    random.seed(1024)
    torch.manual_seed(1024)

    cfg = read_config(os.path.join(conf_path, '{}.yaml'.format(args.config_name)))

    if cfg.model.num_classes == None:
        num_classes = len(os.listdir(multiclass_masks_path))
    else:
        num_classes = cfg.model.num_classes

    data_path = cfg.dataset.data_path
    batch_size = cfg.dataset.batch_size
    num_workers = cfg.opt.num_workers
    cfg.dataset.fold = args.fold

    model_dir = os.path.join(model_results, args.config_name, cfg.model.encoder_name, f"fold_{args.fold}"
                             , cfg.model.exp_name + f'_w_{cfg.opt.weight[0]}_{cfg.opt.weight[1]}_{cfg.opt.weight[2]}'+ "_" + now)

    if "cropped" in data_path:
        print(' cropped data will be used')
        cfg.dataset.cropped = 1
    else:
        cfg.dataset.cropped = 0


    if cfg.opt.processor:
        processor = SegformerImageProcessor.from_pretrained(cfg.model.encoder_name)
    else:
        processor = None
    # opening a file
    with open('./preprocessing/class_mapping.yaml', 'r') as stream:
        try:
            # Converts yaml document to python object
            label2id = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    label2id['bkg'] = 0
    id2label = {v: k for k, v in label2id.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(cfg.model.encoder_name,
                                                             num_labels=num_classes + 1,
                                                             id2label=id2label,
                                                             label2id=label2id,
                                                             ignore_mismatched_sizes=True)

    for param in model.parameters():
        param.requires_grad = False

    params = list(model.named_parameters())
    params.reverse()
    for ix, (name, param) in enumerate(params):
        if ix + 1 <= cfg.opt.from_last_to_unfreeze:  # list(model.named_parameters())[-24][1].requires_grad
            # params_to_update.append(param)          # list(list(model.named_children())[0][1].named_children())
            param.requires_grad = True

    # Set the model in training mode
    model.to(device)

    if cfg.dataset.augmentation:
        if cfg.opt.processor:
            print('train augmentation')
            transform = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                 A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=15, p=0.3),
                 ToTensorV2(),
                 ], additional_targets={'mask':'mask'}
            )
        else:
            if cfg.dataset.normalize_imagenet:
                print('imagenet normalization')
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
            else:
                print('0-1 normalization')
                mean = (0.0, 0.0, 0.0)
                std = (1.0, 1.0, 1.0)
            transform = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                 A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=15, p=0.3),
                 A.Normalize(mean=mean, std=std),
                 ToTensorV2(),
                 ], additional_targets={'mask':'mask'})
    else:
        transform = None

    if cfg.dataset.val_aug:
        print('val augmentation')
        if cfg.opt.processor:
            val_transform = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                 A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=15, p=0.3),
                 ToTensorV2(),
                 ], additional_targets={'mask':'mask'}
            )
        else:
            if cfg.dataset.normalize_imagenet:
                print('imagenet normalization')
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
            else:
                print('0-1 normalization')
                mean = (0.0, 0.0, 0.0)
                std = (1.0, 1.0, 1.0)
            val_transform = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                 A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=15, p=0.3),
                 A.Normalize(mean=mean, std=std),
                 ToTensorV2(),
                 ], additional_targets={'mask': 'mask'}
            )

    else:
        val_transform = None
        print('no val augmentation')

    train_df_path = os.path.join(k_fold_data_path, f'fold_{cfg.dataset.fold}', "train")
    val_df_path = os.path.join(k_fold_data_path, f'fold_{cfg.dataset.fold}', "val")

    train_dataset = KFoldDataframeMulticlassProcessor_v2(data_path=data_path, df_path=train_df_path,
                                   transform=transform, cropped=cfg.dataset.cropped, normalize_imagenet=cfg.dataset.normalize_imagenet,
                            processor=processor)    #,rescale_before_norm=cfg.dataset.rescale_before_norm)
    val_dataset = KFoldDataframeMulticlassProcessor_v2(data_path=data_path, df_path=val_df_path,
                                 transform=val_transform, cropped=cfg.dataset.cropped, normalize_imagenet=cfg.dataset.normalize_imagenet,
                                processor=processor)    #,rescale_before_norm=cfg.dataset.rescale_before_norm)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True)

    if cfg.opt.crossentropy_loss:
        if cfg.opt.weight is not None:
            if cfg.opt.weight == 0:
                raise NotImplementedError
            weight = cfg.opt.weight
            weight = weight if isinstance(weight, torch.FloatTensor) else torch.FloatTensor(weight)
            if cfg.opt.ignore_index is not None:
                criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_idex=cfg.opt.ignore_index).to(
                    device)  # weight (Tensor, optional): a manual rescaling weight given
                # to each class as to be a Tensor of size `C` and floating point dtype
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=weight).to(
                    device)  # weight (Tensor, optional): a manual rescaling weight given
                # to each class as to be a Tensor of size `C` and floating point dtype
                print('crossentropy with pos weights')
        else:
            if cfg.opt.ignore_index is not None:
                criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.opt.ignore_index)
            else:
                criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss()
        print('BCE')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=cfg.opt.es_patience)

    training_cycle_segformer_multiclass(cfg=cfg, model=model, train_loader=train_dataloader,
                                      val_loader=val_dataloader,
                                      criterion=criterion, optimizer=optimizer
                                      , scheduler=scheduler, early_stopping=early_stopping,
                                      model_name=cfg.model.name,
                                      out_dir=model_dir, device=device,
                                      num_epochs=cfg.opt.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--config_name", default='segformer_k_fold', help="Path to the input image")
    parser.add_argument("--fold", default=1, help="Path to the input image")

    args = parser.parse_args()
    main(args)

