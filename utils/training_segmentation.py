"""
Training utilities for segmentation models.
Generic functions for all architectures: DeepLabV3+, PSPNet, U-Net, Mask2Former
"""

import torch
import numpy as np
from tqdm import tqdm
import wandb
import os
from torchmetrics.classification import JaccardIndex

from utils.model_factory import SegmentationModelFactory



def load_segmentation_model(cfg, device):
    """
    Load segmentation model using factory pattern.
    Works for all architectures: DeepLabV3+, PSPNet, U-Net, Mask2Former
    """
    factory = SegmentationModelFactory(cfg, device)
    model = factory.build()

    # Load checkpoint if exists
    if cfg.model.checkpoint and os.path.exists(cfg.model.checkpoint):
        checkpoint = torch.load(cfg.model.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint from {cfg.model.checkpoint}")

    return model


def create_segmentation_dataloader(cfg, batch_size, num_workers):
    """
    Create dataloaders for segmentation.
    Works for all architectures.
    """
    from dataset.segmentation_dataset import SegmentationDataset, segmentation_collate_fn
    from torch.utils.data import DataLoader

    crop_size = cfg.dataset.get('crop_size', 512)

    train_dataset = SegmentationDataset(
        data_path=cfg.dataset.data_path,
        fold=cfg.dataset.fold,
        split='train',
        augmentation=cfg.dataset.augmentation,
        normalize=cfg.dataset.normalize_imagenet,
        crop_size=crop_size
    )

    val_dataset = SegmentationDataset(
        data_path=cfg.dataset.data_path,
        fold=cfg.dataset.fold,
        split='val',
        augmentation=cfg.dataset.val_aug,
        normalize=cfg.dataset.normalize_imagenet,
        crop_size=crop_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=segmentation_collate_fn
    )

    return train_loader, val_loader


def training_cycle_segmentation(cfg, model, train_loader, val_loader, optimizer,
                                scheduler, early_stopping, model_name, out_dir,
                                device, num_epochs, metric_goal):
    """
    Training loop for segmentation models.
    Works for all architectures: DeepLabV3+, PSPNet, U-Net, Mask2Former
    """
    best_metric = float('-inf') if metric_goal == 'maximize' else float('inf')

    # Initialize IoU metrics
    num_classes = cfg.model.num_classes
    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, average="none").to(device)
    iou_metric_mean = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        # ===== TRAINING PHASE =====
        model.train()
        running_loss = 0.0
        running_iou = torch.zeros(num_classes, device=device)
        running_iou_mean = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Training", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            images = images.float()  # Ensure float32

            print(images.max(), torch.unique(batch['targets'][0]['masks']))

            batch_size, _, h, w = images.shape
            semantic_masks = torch.zeros((batch_size, h, w), dtype=torch.long, device=device)

            # Convert instance masks to semantic masks
            for i, target in enumerate(batch['targets']):
                masks = target['masks'].to(device)
                labels = target['labels'].to(device)

                for mask, label in zip(masks, labels):
                    binary_mask = (mask > 0.5).long()
                    semantic_masks[i][binary_mask == 1] = label

            # Prepare data_samples for MMSeg
            data_samples = []
            for i in range(batch_size):
                from mmseg.structures import SegDataSample
                from mmengine.structures import PixelData
                data_sample = SegDataSample()
                gt_sem_seg_data = PixelData()
                gt_sem_seg_data.data = semantic_masks[i]
                data_sample.gt_sem_seg = gt_sem_seg_data
                data_samples.append(data_sample)

            # Forward + backward
            optimizer.zero_grad()

            losses_dict = model(images, data_samples, mode='loss')
            losses = sum([v for k, v in losses_dict.items() if 'loss' in k.lower()])
            losses.backward()
            optimizer.step()

            # Get predictions for IoU
            with torch.no_grad():
                # Add metadata for predict mode
                for data_sample in data_samples:
                    data_sample.set_metainfo({
                        'ori_shape': (h, w),
                        'img_shape': (h, w),
                        'pad_shape': (h, w),
                        'scale_factor': (1.0, 1.0)
                    })

                results = model(images, data_samples, mode='predict')
                preds = torch.stack([result.pred_sem_seg.data for result in results])
                preds = preds.squeeze(1)  # Remove channel dimension

                iou_scores = iou_metric(preds, semantic_masks)
                iou_score_mean = iou_metric_mean(preds, semantic_masks)

            running_loss += losses.item()
            running_iou += iou_scores
            running_iou_mean += iou_score_mean.item()
            train_steps += 1

            pbar.set_postfix(
                loss=running_loss / train_steps,
                iou=(running_iou / train_steps).mean().item()
            )

        # Compute epoch metrics
        train_loss_epoch = running_loss / train_steps
        train_iou_epoch = (running_iou / train_steps).cpu().tolist()
        train_iou_mean_epoch = running_iou_mean / train_steps
        train_losses.append(train_loss_epoch)

        # Log training metrics
        log_dict = {
            "Train Loss": train_loss_epoch,
            "Train IoU": train_iou_mean_epoch,
            "Train IoU Mean": sum(train_iou_epoch) / len(train_iou_epoch),
            "Epoch": epoch + 1
        }
        for c in range(num_classes):
            log_dict[f"Train IoU Class {c}"] = train_iou_epoch[c]
        wandb.log(log_dict)

        # ===== VALIDATION PHASE =====
        model.eval()
        running_loss = 0.0
        running_iou = torch.zeros(num_classes, device=device)
        running_iou_mean = 0.0
        val_steps = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", unit="batch")
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(device)
                images = images.float()

                batch_size, _, h, w = images.shape
                semantic_masks = torch.zeros((batch_size, h, w), dtype=torch.long, device=device)

                for i, target in enumerate(batch['targets']):
                    masks = target['masks'].to(device)
                    labels = target['labels'].to(device)

                    for mask, label in zip(masks, labels):
                        binary_mask = (mask > 0.5).long()
                        semantic_masks[i][binary_mask == 1] = label

                # Prepare data_samples
                data_samples = []
                for i in range(batch_size):
                    from mmseg.structures import SegDataSample
                    from mmengine.structures import PixelData
                    data_sample = SegDataSample()
                    gt_sem_seg_data = PixelData()
                    gt_sem_seg_data.data = semantic_masks[i]
                    data_sample.gt_sem_seg = gt_sem_seg_data
                    data_samples.append(data_sample)

                # Forward
                losses_dict = model(images, data_samples, mode='loss')
                losses = sum([v for k, v in losses_dict.items() if 'loss' in k.lower()])

                # Get predictions
                for data_sample in data_samples:
                    data_sample.set_metainfo({
                        'ori_shape': (h, w),
                        'img_shape': (h, w),
                        'pad_shape': (h, w),
                        'scale_factor': (1.0, 1.0)
                    })

                results = model(images, data_samples, mode='predict')
                preds = torch.stack([result.pred_sem_seg.data for result in results])
                preds = preds.squeeze(1)

                iou_scores = iou_metric(preds, semantic_masks)
                iou_score_mean = iou_metric_mean(preds, semantic_masks)

                running_loss += losses.item()
                running_iou += iou_scores
                running_iou_mean += iou_score_mean.item()
                val_steps += 1

                pbar.set_postfix(
                    loss=running_loss / val_steps,
                    iou=(running_iou / val_steps).mean().item()
                )

        # Compute validation metrics
        val_loss_epoch = running_loss / val_steps
        val_iou_epoch = (running_iou / val_steps).cpu().tolist()
        val_iou_mean_epoch = running_iou_mean / val_steps
        val_losses.append(val_loss_epoch)

        # Log validation metrics
        log_dict = {
            "Validation Loss": val_loss_epoch,
            "Validation IoU": val_iou_mean_epoch,
            "Validation IoU Mean": sum(val_iou_epoch) / len(val_iou_epoch),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "Epoch": epoch + 1
        }
        for c in range(num_classes):
            log_dict[f"Validation IoU Class {c}"] = val_iou_epoch[c]
        wandb.log(log_dict)

        print(
            f'\nEpoch {epoch + 1}: Val Loss = {val_loss_epoch:.4f}, Val IoU = {sum(val_iou_epoch) / len(val_iou_epoch):.4f}')
        print(f"Per-class IoU: {val_iou_epoch}")

        scheduler.step(val_loss_epoch)
        early_stopping(val_loss_epoch)

        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
            break

        # Save best model
        save_condition = (metric_goal == "minimize" and val_loss_epoch < best_metric) or \
                         (metric_goal == "maximize" and sum(val_iou_epoch) / len(val_iou_epoch) > best_metric)

        if save_condition:
            best_metric = val_loss_epoch if metric_goal == "minimize" else sum(val_iou_epoch) / len(val_iou_epoch)
            print(f'\n✓ Validation {metric_goal} improved, saving model...')

            os.makedirs(out_dir, exist_ok=True)
            checkpoint_path = os.path.join(out_dir, f'{model_name}.pth')

            model_to_save = model.module if hasattr(model, 'module') else model

            torch.save({
                'cfg': cfg,
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_epoch,
                'val_iou': sum(val_iou_epoch) / len(val_iou_epoch),
                'train_loss_history': train_losses,
                'val_loss_history': val_losses,
            }, checkpoint_path)

            wandb.run.summary["best_val_iou"] = sum(val_iou_epoch) / len(val_iou_epoch)
            wandb.run.summary["best_epoch"] = epoch + 1

    print(f"\n{'=' * 50}")
    print(f"🎉 Training completed!")
    print(f"   Best metric: {best_metric:.4f}")
    print(f"{'=' * 50}")

    wandb.finish()