import argparse
from torch.utils.data import DataLoader
from dataset.segmentation import SemanticSegmentationDataset, BinarySegmentationAlb, BinarySegmentationPil
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (RandomCrop,CenterCrop,ElasticTransform,RGBShift,Rotate,
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,Transpose,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip, HorizontalFlip,
    HueSaturationValue,
)
import numpy as np
import torch
import torch.optim
import segmentation_models_pytorch as smp
from utils.training import training_cycle, training_cycle_deeplab
from utils.opt import EarlyStopping
import yaml
import json
import types
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import datetime
from torch.utils.data import ConcatDataset, Dataset
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

    random_seed = cfg.dataset.random_seed
    shuffle = cfg.dataset.shuffle
    validation_split = cfg.dataset.validation_split
    data_path = cfg.dataset.data_path
    val_data_path = cfg.dataset.val_data_path
    batch_size = cfg.dataset.batch_size
    num_workers = cfg.opt.num_workers

    if args.config_name == 'resnet':
        model = smp.Unet(cfg.model.encoder_name).to(device)  # By default activation is none
        params = smp.encoders.get_preprocessing_params(cfg.model.encoder_name)

        for param in model.parameters():
            param.requires_grad = False

        params = list(model.named_parameters())
        params.reverse()
        for ix, (name, param) in enumerate(params):
            if ix + 1 <= cfg.opt.from_last_to_unfreeze:   #list(model.named_parameters())[-24][1].requires_grad
                # params_to_update.append(param)          # list(list(model.named_children())[0][1].named_children())
                param.requires_grad = True

    elif args.config_name == 'c-resunet':
        model = c_resunet(arch='c-ResUnet', n_features_start=cfg.model.n_features_start, n_out=1,
                          pretrained=False, progress=True).to(device)
        encoder_name = cfg.model.encoder_name
    elif args.config_name == 'deeplab':
        model = torch.hub.load('pytorch/vision:v0.10.0', cfg.model.encoder_name, pretrained=True).to(device)
        #model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

        if cfg.model.remove_aux:
            model.aux_classifier = None
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = DeepLabHead(2048, num_classes=1)

        params = list(model.named_parameters())
        params.reverse()
        for ix, (name, param) in enumerate(params):
            if ix + 1 <= cfg.opt.from_last_to_unfreeze:   #list(model.named_parameters())[-24][1].requires_grad
                # params_to_update.append(param)          # list(list(model.named_children())[0][1].named_children())
                param.requires_grad = True

        encoder_name = cfg.model.encoder_name
    elif args.config_name == 'segformer':
        raise NotImplementedError

    # Set the model in training mode
    model.to(device)

    if cfg.dataset.normalize_imagenet:
        print('imagenet normalization')
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif cfg.dataset.automatic_normalize:
        std = torch.tensor(params["std"]).view(1, 3, 1, 1)
        mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
        std = tuple(std.squeeze().tolist())
        mean = tuple(mean.squeeze().tolist())
        print('imagenet normalization')
    else:
        print('0-1 normalization')
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)

    if cfg.dataset.augmentation:
        print('train augmentation')

        transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=mean, std=std),
                OneOf([
                    A.VerticalFlip(p=0.3),
                    A.HorizontalFlip(p=0.3),
                ], p=0.6),
                OneOf([
                    #A.CLAHE(p=0.3),
                    A.ImageCompression(quality_lower=75, p=0.2),
                    Blur(blur_limit=21, p=0.4),
                ], p=0.5),

                # ElasticTransform(alpha=60, sigma=8, alpha_affine=60, interpolation=1, border_mode=4,
                #             always_apply=False, approximate=False,
                #             p=0.2),
                ToTensorV2(),
            ]
                )
    else:
        transform = None
        print('no train augmentation')

    if cfg.dataset.val_aug:
        print('val augmentation')
        val_transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
                A.VerticalFlip(p=0.2),
                A.HorizontalFlip(p=0.2),

                Blur(blur_limit=15, p=0.3),

                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        val_transform = None
        print('no val augmentation')

    images_path = os.path.join(data_path, 'images')
    val_images_path = os.path.join(val_data_path, 'images')

    if cfg.dataset.merge_train_val:

        dataset_size = len(os.listdir(images_path))
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices_train, val_indices_train = indices[split:], indices[:split]

        val_dataset_size = len(os.listdir(val_images_path))
        indices = list(range(val_dataset_size))
        split = int(np.floor(validation_split * val_dataset_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices_val, val_indices_val = indices[split:], indices[:split]

        if cfg.dataset.pil_version:
            train_dataset_train = BinarySegmentationPil(data_path, idxs=train_indices_train, transform=None)
            train_dataset_val = BinarySegmentationPil(val_data_path, idxs=train_indices_val, transform=None)
            val_dataset_train = BinarySegmentationPil(data_path, idxs=val_indices_train, transform=None)
            val_dataset_val = BinarySegmentationPil(val_data_path, idxs=val_indices_val, transform=None)
            train_dataset = ConcatDataset([train_dataset_train, train_dataset_val])
            val_dataset = ConcatDataset([val_dataset_train, val_dataset_val])
        else:
            if cfg.dataset.augmentation:
                train_dataset_train = BinarySegmentationAlb(data_path, idxs=train_indices_train,
                                                            transform=transform)
                train_dataset_val = BinarySegmentationAlb(val_data_path, idxs=train_indices_val, transform=transform)
                val_dataset_train = BinarySegmentationAlb(data_path, idxs=val_indices_train,
                                                          transform=val_transform)
                val_dataset_val = BinarySegmentationAlb(val_data_path, idxs=val_indices_val, transform=val_transform)

                if cfg.dataset.add_base_images:
                    train_dataset_train_base = BinarySegmentationAlb(data_path, idxs=train_indices_train,
                                                                transform=None)
                    train_dataset_val_base = BinarySegmentationAlb(val_data_path, idxs=train_indices_val, transform=None)
                    val_dataset_train_base = BinarySegmentationAlb(data_path, idxs=val_indices_train,
                                                              transform=None)
                    val_dataset_val_base = BinarySegmentationAlb(val_data_path, idxs=val_indices_val, transform=None)

                    train_dataset = ConcatDataset([train_dataset_train, train_dataset_val
                                                      , train_dataset_train_base, train_dataset_val_base])
                    val_dataset = ConcatDataset([val_dataset_train, val_dataset_val
                                                    , val_dataset_train_base, val_dataset_val_base])
                else:
                    train_dataset = ConcatDataset([train_dataset_train, train_dataset_val])
                    val_dataset = ConcatDataset([val_dataset_train, val_dataset_val])
    else:

        if cfg.dataset.pil_version:
            print(' with pil version augmentation is not implemented')
            print(' with pil version augmentation is not implemented')
            print(' with pil version augmentation is not implemented')

            train_dataset = BinarySegmentationPil(data_path, transform=None)
            val_dataset = BinarySegmentationPil(val_data_path, transform=None)
        else:
            if cfg.dataset.augmentation:

                train_dataset = BinarySegmentationAlb(data_path, transform=transform)
                val_dataset = BinarySegmentationAlb(val_data_path, transform=val_transform)

                if cfg.dataset.add_base_images:
                    train_dataset_base = BinarySegmentationAlb(data_path, transform=None)
                    val_dataset_base = BinarySegmentationAlb(val_data_path, transform=None)

                    train_dataset = ConcatDataset([train_dataset, train_dataset_base])
                    val_dataset = ConcatDataset([val_dataset, val_dataset_base])

            else:
                train_dataset = BinarySegmentationAlb(data_path, transform=transform)
                val_dataset = BinarySegmentationAlb(val_data_path, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)

    if cfg.opt.logit_loss:
        if cfg.opt.pos_weight is not None:
            if cfg.opt.pos_weight == 0:
                raise NotImplementedError
            weight_pos = cfg.opt.pos_weight
            weight_pos = weight_pos if isinstance(weight_pos, torch.FloatTensor) else torch.FloatTensor([weight_pos])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_pos]).to(device))
            print('logit loss with pos weights')
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            print('logit loss')
    else:
        criterion = torch.nn.BCELoss()
        print('BCE')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=cfg.opt.es_patience)
    model_dir = os.path.join(model_results, args.config_name, cfg.model.encoder_name, cfg.model.exp_name + "_" + now)

    if 'deeplab' in args.config_name:
        training_cycle_deeplab(cfg=cfg, model=model, train_loader=train_dataloader, val_loader=val_dataloader,
                                   criterion=criterion, optimizer=optimizer
                                   , scheduler=scheduler, early_stopping=early_stopping, model_name=cfg.model.name,
                                   out_dir=model_dir, device=device,
                                   num_epochs=cfg.opt.epochs)
    else:
        training_cycle(cfg=cfg, model=model, train_loader=train_dataloader, val_loader=val_dataloader, criterion=criterion,
                           optimizer=optimizer
                           , scheduler=scheduler, early_stopping=early_stopping,
                           model_name=cfg.model.name,
                           out_dir=model_dir, device=device,
                           num_epochs=cfg.opt.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop image and update annotation")
    parser.add_argument("--config_name", default='deeplab', help="Path to the input image")

    args = parser.parse_args()
    main(args)

