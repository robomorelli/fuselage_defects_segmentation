import torch
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss
import numpy as np
from torch import nn
import wandb
import yaml
from dataset.segmentation import KFoldDataframeMulticlassProcessor
from torchmetrics.classification import JaccardIndex
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
    SegformerImageProcessor)
from torch.utils.data import DataLoader
from config import *

def training_cycle(cfg, model, train_loader, val_loader, criterion, optimizer,
                       scheduler, early_stopping, model_name='cnn', out_dir=model_results, device='cpu', num_epochs=200):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    val_loss = 10 ** 16
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, masks) in enumerate(tepoch):

                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()
                if not cfg.opt.logit_loss:
                    if cfg.model.activation != None:
                        outputs = model(inputs.to(device))
                    else:
                        outputs = model(inputs.to(device)).sigmoid()
                else:
                    outputs = model(inputs.to(device))

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss.item()/(i+1))

            # Print average training loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")
            train_losses.append(running_loss / len(train_loader))

            # Validation loop
            model.eval()
            running_loss = 0.0

            with torch.no_grad():
                with tqdm(val_loader, unit="batch") as vepoch:
                    for i, (inputs, masks) in enumerate(vepoch):

                        inputs, masks = inputs.to(device), masks.to(device)

                        if not cfg.opt.logit_loss:
                            if cfg.model.activation != None:
                                outputs = model(inputs.to(device))
                            else:
                                outputs = model(inputs.to(device)).sigmoid()
                        else:
                            outputs = model(inputs.to(device))

                        loss += criterion(outputs, masks).item()
                        running_loss += loss

                        vepoch.set_postfix(loss=running_loss/(i+1))

                val_loss_epoch = running_loss / len(val_loader)
                val_losses.append(val_loss_epoch)

                scheduler.step(val_loss_epoch)
                print('eval loss {}'.format(val_loss_epoch))
                early_stopping(val_loss_epoch)
                if early_stopping.early_stop:
                    break
                if val_loss_epoch < val_loss:
                    print('val_loss improved from {} to {}, saving model  {} to {}' \
                          .format(val_loss, val_loss_epoch, model_name, out_dir))
                    torch.save({
                        'cfg': cfg,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss_epoch,
                        'train_loss_history': train_losses,
                        'val_loss_history': val_losses,
                    }, out_dir + '/{}.pth'.format(model_name))
                    val_loss = val_loss_epoch


def training_cycle_segformer_multiclass(cfg, model, train_loader, val_loader, criterion, optimizer,
                                        scheduler, early_stopping, model_name='cnn',
                                        out_dir='model_results', device='cpu', num_epochs=200, metric_goal="maximize"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if metric_goal == "minimize":
        best_metric = float("inf")  # Start with a high value for minimization
    else:
        best_metric = float("-inf")  # Start with a low value for maximization

    # Do a metric class that based on the name instntiate the right object
    # Do a metric class that based on the name instntiate the right object
    # Do a metric class that based on the name instntiate the right object
    # Initialize IoU metric
    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes+1).to(device)

    val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_iou = 0.0, 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, masks) in enumerate(tepoch):
                optimizer.zero_grad()

                pixel_values = inputs.to(device)
                labels = masks.to(device)

                outputs = model(pixel_values=pixel_values, labels=labels.long()).logits
                print(f"Has nan {torch.isnan(labels).any()} has Nan {torch.isnan(outputs).any()}")
                print(f"Has nan {np.unique(labels.detach().cpu())} has Nan {np.unique(outputs.detach().cpu())}")
                upsampled_logits = nn.functional.interpolate(
                    outputs,
                    size=tuple(inputs.shape[-2:]),
                    mode='bilinear',
                    align_corners=False
                )

                loss = criterion(upsampled_logits.float(), labels.squeeze(1).long())
                loss.backward()
                optimizer.step()

                preds = torch.argmax(upsampled_logits, dim=1)
                iou_score = iou_metric(preds, labels.squeeze(1))

                running_loss += loss.item()
                running_iou += iou_score.item()

                tepoch.set_postfix(loss=running_loss / (i + 1), iou=running_iou / (i + 1))

            train_loss_epoch = running_loss / len(train_loader)
            train_iou_epoch = running_iou / len(train_loader)
            train_losses.append(train_loss_epoch)

            # this is the log dict logging
            wandb.log({"Train Loss": train_loss_epoch, "Train IoU": train_iou_epoch, "Epoch": epoch})

        model.eval()
        running_loss, running_iou = 0.0, 0.0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vepoch:
                for i, (inputs, masks) in enumerate(vepoch):
                    pixel_values = inputs.to(device)
                    labels = masks.to(device)

                    #print(np.any(np.isnan(pixel_values.detach().cpu())), np.any(np.isnan(labels.detach().cpu())))

                    outputs = model(pixel_values=pixel_values, labels=labels.long()).logits
                    upsampled_logits = nn.functional.interpolate(
                        outputs,
                        size=tuple(inputs.shape[-2:]),
                        mode='bilinear',
                        align_corners=False
                    )

                    loss = criterion(upsampled_logits.float(), labels.squeeze(1).long())
                    preds = torch.argmax(upsampled_logits, dim=1)
                    iou_score = iou_metric(preds, labels.squeeze(1))

                    running_loss += loss.item()
                    running_iou += iou_score.item()

                    vepoch.set_postfix(loss=running_loss / (i + 1), iou=running_iou / (i + 1))

                val_loss_epoch = running_loss / len(val_loader)
                val_iou_epoch = running_iou / len(val_loader)
                val_losses.append(val_loss_epoch)

                wandb.log({"Validation Loss": val_loss_epoch, "Validation IoU": val_iou_epoch, "Epoch": epoch})

            scheduler.step(val_loss_epoch)
            print(f'Epoch {epoch + 1}: Val Loss = {val_loss_epoch}, Val IoU = {val_iou_epoch}')

            early_stopping(val_loss_epoch)
            if early_stopping.early_stop:
                break

            save_condition = (metric_goal == "minimize" and val_loss_epoch < best_metric) or \
                             (metric_goal == "maximize" and val_iou_epoch > best_metric)

            if save_condition:
                best_metric = val_loss_epoch if metric_goal == "minimize" else val_iou_epoch
                print(f'Validation {metric_goal} improved, saving model...')
                torch.save({
                    'cfg': cfg,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_epoch,
                    'train_loss_history': train_losses,
                    'val_loss_history': val_losses,
                }, os.path.join(out_dir, f'{model_name}.pth'))

                save_checkpoint_wandb(path=cfg.model.checkpoint)

    wandb.finish()


def training_cycle_segformer_multiclass_bkp(cfg, model, train_loader, val_loader, criterion, optimizer,
                       scheduler, early_stopping, model_name='cnn'
                           , out_dir=model_results, device='cpu', num_epochs=200):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    val_loss = 10 ** 16
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, masks) in enumerate(tepoch):

                print(np.unique(masks.cpu()))


                # get the inputs;
                pixel_values = inputs.to(device)
                labels = masks.to(device)

                outputs = model(pixel_values=pixel_values, labels=labels.long()).logits
                #loss, logits = outputs.loss, outputs.logits

                optimizer.zero_grad()
                #outputs = model(inputs.squeeze()).logits
                upsampled_logits = nn.functional.interpolate(
                    outputs,
                    size=tuple(inputs.shape[-2:]),  # (height, width)
                    mode='bilinear',
                    align_corners=False
                )
                if cfg.opt.crossentropy_loss:
                    loss = criterion(upsampled_logits.float(), labels.squeeze(1).long())
                else:
                    raise NotImplementedError

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss/(i+1)) #,dice_loss=running_dice_loss/(i+1))

            # Print average training loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")
            train_losses.append(running_loss / len(train_loader))

        # Validation loop
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vepoch:
                for i, (inputs, masks) in enumerate(vepoch):

                    # get the inputs;
                    pixel_values = inputs.to(device)
                    labels = masks.to(device)

                    #print(pixel_values.min(), pixel_values.max())

                    # if 1 in list(np.unique(masks.cpu())) or 2 in list(np.unique(masks.cpu())):
                    #    print('mark or graffio')

                    outputs = model(pixel_values=pixel_values, labels=labels.long()).logits
                    # loss, logits = outputs.loss, outputs.logits

                    # outputs = model(inputs.squeeze()).logits
                    upsampled_logits = nn.functional.interpolate(
                        outputs,
                        size=tuple(inputs.shape[-2:]),  # (height, width)
                        mode='bilinear',
                        align_corners=False
                    )

                    #inputs, masks = inputs.to(device), masks.to(device)
                    #outputs = model(inputs.squeeze()).logits

                    if cfg.opt.crossentropy_loss:
                        #one_channel_out = torch.argmax(outputs['out'].softmax(1), 1)
                        loss = criterion(upsampled_logits.float(), labels.squeeze(1).long())
                    else:
                        raise NotImplementedError

                    running_loss += loss.item()
                    #running_dice_loss += dice_loss.item()

                    vepoch.set_postfix(loss=running_loss/(i+1))#, dice_loss=running_dice_loss/(i+1))

            val_loss_epoch = running_loss / len(val_loader)
            #val_dice_loss_epoch = running_dice_loss / len(val_loader)
            val_losses.append(val_loss_epoch)

            scheduler.step(val_loss_epoch)
            print('eval loss {} '.format(val_loss_epoch))#, val_dice_loss_epoch))

            early_stopping(val_loss_epoch)
            if not cfg.opt.save_each_epoch:

                if early_stopping.early_stop:
                    break
                if val_loss_epoch < val_loss:
                    print('val_loss improved from {} to {}, saving model  {} to {}' \
                          .format(val_loss, val_loss_epoch, model_name, out_dir))
                    torch.save({
                        'cfg': cfg,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss_epoch,
                        'train_loss_history': train_losses,
                        'val_loss_history': val_losses,
                    }, out_dir + '/{}.pth'.format(model_name))
                    val_loss = val_loss_epoch
            else:
                print('val_loss from {} to {}, saving model  {} to {}' \
                      .format(val_loss, val_loss_epoch, model_name, out_dir))
                torch.save({
                    'cfg': cfg,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_epoch,
                    'train_loss_history': train_losses,
                    'val_loss_history': val_losses,
                }, out_dir + '/{}.pth'.format(model_name))
                val_loss = val_loss_epoch



def training_cycle_segformer_multiclass_implicit_loss(cfg, model, train_loader, val_loader, criterion, optimizer,
                       scheduler, early_stopping, model_name='cnn'
                           , out_dir=model_results, device='cpu', num_epochs=200):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    val_loss = 10 ** 16
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (image, mask) in enumerate(tepoch):

                optimizer.zero_grad()

                # get the inputs;
                pixel_values = image.to(device)
                labels = mask.to(device)

                #if 1 in list(np.unique(labels.cpu())) or 2 in list(np.unique(labels.cpu())):
                #    print('mark or graffio')

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss, logits = outputs.loss, outputs.logits

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss/(i+1)) #,dice_loss=running_dice_loss/(i+1))

            # Print average training loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")
            train_losses.append(running_loss / len(train_loader))

        # Validation loop
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vepoch:
                for i, (image, mask) in enumerate(vepoch):

                    # get the inputs;
                    pixel_values = image.to(device)
                    labels = mask.to(device)

                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss, logits = outputs.loss, outputs.logits

                    running_loss += loss.item()

                    vepoch.set_postfix(loss=running_loss/(i+1))#, dice_loss=running_dice_loss/(i+1))

            val_loss_epoch = running_loss / len(val_loader)
            val_losses.append(val_loss_epoch)

            scheduler.step(val_loss_epoch)
            print('eval loss {} '.format(val_loss_epoch))#, val_dice_loss_epoch))

            early_stopping(val_loss_epoch)
            if not cfg.opt.save_each_epoch:

                if early_stopping.early_stop:
                    break
                if val_loss_epoch < val_loss:
                    print('val_loss improved from {} to {}, saving model  {} to {}' \
                          .format(val_loss, val_loss_epoch, model_name, out_dir))
                    torch.save({
                        'cfg': cfg,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss_epoch,
                        'train_loss_history': train_losses,
                        'val_loss_history': val_losses,
                    }, out_dir + '/{}.pth'.format(model_name))
                    val_loss = val_loss_epoch
            else:
                print('val_loss from {} to {}, saving model  {} to {}' \
                      .format(val_loss, val_loss_epoch, model_name, out_dir))
                torch.save({
                    'cfg': cfg,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_epoch,
                    'train_loss_history': train_losses,
                    'val_loss_history': val_losses,
                }, out_dir + '/{}.pth'.format(model_name))
                val_loss = val_loss_epoch

def load_model(cfg, layers_to_unfreeze=10000):

    with open(class_mapping_outfile, 'r') as stream:
        try:
            # Converts yaml document to python object
            label2id = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    label2id['bkg'] = 0
    id2label = {v: k for k, v in label2id.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(cfg.model.encoder_name,
                                                             num_labels=num_classes + 1,
                                                             id2label=id2label, label2id=label2id,
                                                             ignore_mismatched_sizes=True)

    for param in model.parameters():
        param.requires_grad = False

    params = list(model.named_parameters())
    params.reverse()
    for ix, (name, param) in enumerate(params):
        if ix + 1 <= layers_to_unfreeze:  # list(model.named_parameters())[-24][1].requires_grad
            # params_to_update.append(param)          # list(list(model.named_children())[0][1].named_children())
            param.requires_grad = True

    return model

def create_dataloader(cfg, batch_size=8, num_workers=0):

    if cfg.opt.processor:
        processor = SegformerImageProcessor.from_pretrained(cfg.model.encoder_name)
    else:
        processor = None
        # opening a file

    if cfg.dataset.augmentation:
        if cfg.opt.processor:
            print('train augmentation')
            transform = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                 A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=19, p=0.35),
                 ToTensorV2(),
                 ], additional_targets={'mask':'mask'}
            )
            cfg.dataset.normalize_imagenet = 0
            cfg.dataset.automatic_normalize = 0
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
                 A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=19, p=0.3),
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
                 A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.HorizontalFlip(p=0.2),
                 Blur(blur_limit=19, p=0.3),
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

    train_dataset = KFoldDataframeMulticlassProcessor(data_path=cfg.dataset.data_path, df_path=train_df_path,
                                   transform=transform, normalize_imagenet=cfg.dataset.normalize_imagenet,
                            processor=processor)    #,rescale_before_norm=cfg.dataset.rescale_before_norm)
    val_dataset = KFoldDataframeMulticlassProcessor(data_path=cfg.dataset.data_path, df_path=val_df_path,
                                 transform=val_transform, normalize_imagenet=cfg.dataset.normalize_imagenet,
                                processor=processor)    #,rescale_before_norm=cfg.dataset.rescale_before_norm)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True)

    return train_dataloader, val_dataloader


def on_fit_epoch_end(trainer, save_model=False, path="model_epoch.pth"):
    """Callback to rename files after training ends."""
    # Get the directory where the model saves files
    #model_dir = trainer.save_dir  # The directory where best.pt and last.pt are saved
    #rename_model_files(model_dir, time_str)
    wandb.log({#**trainer.lr,
               **trainer.metrics})

    if save_model:
        # Upload checkpoint to W&B
        artifact = wandb.Artifact(f"model", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)


def save_checkpoint_wandb(path="model.pth"):

    # Upload checkpoint to W&B
    artifact = wandb.Artifact(f"model", type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)

