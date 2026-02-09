"""
Evaluation script for MMSeg-based segmentation models.
Compatible with: DeepLabV3+, PSPNet, SegFormer, Mask2Former
"""

import os
import sys

# Fix Qt/xcb errors - MUST BE BEFORE ANY OTHER IMPORTS
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

# Now safe to import matplotlib
import matplotlib
matplotlib.use('Agg')

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import random
from torchmetrics.classification import JaccardIndex

# Add parent directory to path
sys.path.append('.')

from utils.model_factory import SegmentationModelFactory
from dataset.segmentation_dataset import EvaluationSegmentationDataset, evaluation_collate_fn


def create_color_map(num_classes):
    """Create colormap for visualization"""
    colors = sns.color_palette("husl", num_classes)
    return ListedColormap(colors)


def visualize_comparison(image, gt_mask, pred_mask, name, save_path, id2label, colormap):
    """
    Create 4-panel visualization:
    - Original image
    - Ground Truth
    - Prediction
    - Overlay (GT in red, Pred in green, correct in yellow)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original image
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed
        if img_np.min() < 0:
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = image

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    # Ground Truth
    gt_colored = axes[0, 1].imshow(gt_mask, cmap=colormap, vmin=0, vmax=len(id2label) - 1)
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')

    # Prediction
    pred_colored = axes[1, 0].imshow(pred_mask, cmap=colormap, vmin=0, vmax=len(id2label) - 1)
    axes[1, 0].set_title('Prediction', fontsize=14)
    axes[1, 0].axis('off')

    # Error overlay
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    # Red: GT but not predicted (False Negative)
    overlay[(gt_mask > 0) & (pred_mask == 0)] = [255, 0, 0]
    # Green: Predicted but not GT (False Positive)
    overlay[(gt_mask == 0) & (pred_mask > 0)] = [0, 255, 0]
    # Yellow: Correctly predicted (True Positive)
    overlay[(gt_mask > 0) & (pred_mask > 0) & (gt_mask == pred_mask)] = [255, 255, 0]
    # White: Wrong class predicted
    overlay[(gt_mask > 0) & (pred_mask > 0) & (gt_mask != pred_mask)] = [255, 255, 255]

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Error Overlay\n(Red=FN, Green=FP, Yellow=TP, White=Wrong Class)',
                         fontsize=12)
    axes[1, 1].axis('off')

    # Add colorbar with class labels
    cbar = plt.colorbar(pred_colored, ax=axes[1, 0], orientation='horizontal',
                        pad=0.05, fraction=0.046)
    cbar.set_ticks(range(len(id2label)))
    cbar.set_ticklabels([id2label[i] for i in range(len(id2label))], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def compute_per_image_metrics(gt_mask, pred_mask, num_classes):
    """Calculate metrics for single image"""
    metrics = {}

    # IoU per class
    ious = []
    for cls in range(num_classes):
        gt_cls = (gt_mask == cls)
        pred_cls = (pred_mask == cls)

        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()

        if union > 0:
            iou = intersection / union
            ious.append(iou)

    metrics['mIoU'] = np.mean(ious) if ious else 0.0
    metrics['per_class_IoU'] = ious

    # Pixel Accuracy
    metrics['pixel_acc'] = (gt_mask == pred_mask).sum() / gt_mask.size

    return metrics


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"\nLoading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config from checkpoint
    cfg = checkpoint.get('cfg', None)
    if cfg is None:
        raise ValueError("Checkpoint does not contain 'cfg'. Cannot rebuild model.")

    print(f"Model architecture: {cfg.model.architecture}")
    print(f"Backbone: {cfg.model.backbone}")
    print(f"Number of classes: {cfg.model.num_classes}")

    # Build model using factory
    factory = SegmentationModelFactory(cfg, device)
    model = factory.build()

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("✓ Model loaded successfully")

    return model, cfg


def create_dataloader(data_path, csv_path, cfg, batch_size=1, cropped=True):
    """
    Create dataloader for evaluation.

    Args:
        data_path: Path to images directory
        csv_path: Path to CSV file or directory containing CSV
        cfg: Config from checkpoint
        batch_size: Batch size
        cropped: If True, look for cropped_filenames.csv, else full_size_filenames.csv
    """
    normalize = cfg.dataset.get('normalize_imagenet', 0)

    if csv_path is not None:
        # Load from CSV (can be file or directory)
        print(f"\nLoading dataset from CSV path: {csv_path}")
        dataset = EvaluationSegmentationDataset(
            data_path=data_path,
            csv_path=csv_path,
            normalize=normalize,
            from_csv=True,
            cropped=cropped
        )
    else:
        # Load from folder
        print(f"\nLoading dataset from folder: {data_path}")
        dataset = EvaluationSegmentationDataset(
            data_path=data_path,
            normalize=normalize,
            from_folder=True,
            cropped=cropped
        )

    print(f"✓ Loaded {len(dataset)} images")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=evaluation_collate_fn
    )

    return dataloader, dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMSeg segmentation models")
    parser.add_argument("--model_path", default="./fuselage_segmentation_hpo_pspnet/57a4d1ug/pspnet/resnet50/fold_1/pspnet_hpo_w_1_1_1.05_09_02_2026_14_25_01/robust-sweep-1/pspnet_resnet50_fold1.pth",
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument("--data_path", default="data/cropped_data",
                        help="Path to data directory (containing images/ and masks/)")
    parser.add_argument("--csv_path", default="data/k-fold/fold_1/test",
                        help="Optional: Path to CSV file or directory with image filenames")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as model checkpoint)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--viz_percentage", type=float, default=100.0,
                        help="Percentage of images to save visualizations (0-100)")
    parser.add_argument("--cropped", type=int, default=1, choices=[0, 1],
                        help="Use cropped_filenames.csv (1) or full_size_filenames.csv (0)")
    parser.add_argument("--device", type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help="Device to use")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    if args.csv_path and not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV path not found: {args.csv_path}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    model, cfg = load_model_from_checkpoint(args.model_path, device)
    model = model.to(device)
    model.eval()

    num_classes = cfg.model.num_classes

    # Create output directory
    if args.output_dir is None:
        model_dir = Path(args.model_path).parent
        csv_name = Path(args.csv_path).stem if args.csv_path else "folder"
        output_dir = model_dir / f"evaluation_{csv_name}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Create subdirectories
    pred_dir = output_dir / "predictions"
    pred_viz_dir = output_dir / "predictions_viz"
    comparison_dir = output_dir / "comparisons"

    pred_dir.mkdir(exist_ok=True)
    pred_viz_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)

    # Create dataloader
    dataloader, dataset = create_dataloader(
        args.data_path,
        args.csv_path,
        cfg,
        args.batch_size,
        cropped=bool(args.cropped)
    )

    # Get filenames
    filenames = dataset.images_file_names

    # Create colormap
    colormap = create_color_map(num_classes)

    # Class labels (create simple mapping)
    id2label = {i: f"Class_{i}" for i in range(num_classes)}

    # Determine which images to visualize
    total_images = len(dataset)
    num_to_visualize = int(total_images * args.viz_percentage / 100)

    if num_to_visualize < total_images:
        viz_indices = set(random.sample(range(total_images), num_to_visualize))
        print(f"\nWill save visualizations for {num_to_visualize}/{total_images} images ({args.viz_percentage}%)")
    else:
        viz_indices = set(range(total_images))
        print(f"\nWill save visualizations for all {total_images} images")

    # Initialize metrics
    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, average="none").to(device)
    iou_metric_mean = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

    all_metrics = []
    all_ious = []

    print("\n" + "=" * 50)
    print("Starting evaluation...")
    print("=" * 50)

    with torch.no_grad():
        current_idx = 0

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = batch['images'].to(device).float()
            batch_size, _, h, w = images.shape

            # Convert targets to semantic masks
            semantic_masks = torch.zeros((batch_size, h, w), dtype=torch.long, device=device)

            for i, target in enumerate(batch['targets']):
                masks = target['masks'].to(device)
                labels = target['labels'].to(device)

                for mask, label in zip(masks, labels):
                    binary_mask = (mask > 0.5).long()
                    semantic_masks[i][binary_mask == 1] = label

            # Prepare data_samples for prediction
            from mmseg.structures import SegDataSample
            from mmengine.structures import PixelData

            data_samples = []
            for i in range(batch_size):
                data_sample = SegDataSample()
                data_sample.set_metainfo({
                    'ori_shape': (h, w),
                    'img_shape': (h, w),
                    'pad_shape': (h, w),
                    'scale_factor': (1.0, 1.0)
                })

                # Add GT for reference
                gt_sem_seg_data = PixelData()
                gt_sem_seg_data.data = semantic_masks[i]
                data_sample.gt_sem_seg = gt_sem_seg_data

                data_samples.append(data_sample)

            # Get predictions
            results = model(images, data_samples, mode='predict')
            preds = torch.stack([result.pred_sem_seg.data for result in results])
            preds = preds.squeeze(1)  # Remove channel dimension

            # Calculate metrics
            iou_scores = iou_metric(preds, semantic_masks)
            iou_score_mean = iou_metric_mean(preds, semantic_masks)

            # Process each image in batch
            for i in range(batch_size):
                img_idx = current_idx + i

                if img_idx >= len(filenames):
                    break

                name = os.path.splitext(filenames[img_idx])[0]

                pred_mask = preds[i].cpu().numpy()
                gt_mask = semantic_masks[i].cpu().numpy()

                # Calculate per-image metrics
                metrics = compute_per_image_metrics(gt_mask, pred_mask, num_classes)
                all_metrics.append({
                    'filename': filenames[img_idx],
                    'mIoU': metrics['mIoU'],
                    'pixel_acc': metrics['pixel_acc']
                })
                all_ious.extend(metrics['per_class_IoU'])

                # Save raw prediction
                cv2.imwrite(str(pred_dir / f"{name}.png"), pred_mask.astype(np.uint8))

                # Save colored visualization
                pred_viz = (pred_mask / num_classes * 255).astype(np.uint8)
                cv2.imwrite(str(pred_viz_dir / f"{name}.png"), pred_viz)

                # Save comparison visualization (only for selected images)
                if img_idx in viz_indices:
                    visualize_comparison(
                        images[i].cpu(),
                        gt_mask,
                        pred_mask,
                        name,
                        comparison_dir,
                        id2label,
                        colormap
                    )

            current_idx += batch_size

    # Save per-image metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / 'per_image_metrics.csv', index=False)

    # Calculate global metrics
    global_metrics = {
        'mean_mIoU': metrics_df['mIoU'].mean(),
        'std_mIoU': metrics_df['mIoU'].std(),
        'mean_pixel_acc': metrics_df['pixel_acc'].mean(),
        'std_pixel_acc': metrics_df['pixel_acc'].std(),
        'overall_mIoU': np.mean(all_ious) if all_ious else 0.0
    }

    # Print results
    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    for key, value in global_metrics.items():
        print(f'{key}: {value:.4f}')
    print('=' * 50)

    # Save global metrics
    with open(output_dir / 'global_metrics.txt', 'w') as f:
        f.write(f'Evaluation on: {args.data_path}\n')
        if args.csv_path:
            f.write(f'CSV file: {args.csv_path}\n')
        f.write(f'Model: {args.model_path}\n')
        f.write(f'Architecture: {cfg.model.architecture}\n')
        f.write(f'Backbone: {cfg.model.backbone}\n')
        f.write(f'Number of classes: {num_classes}\n')
        f.write(f'Total images: {total_images}\n')
        f.write(f'Visualizations saved: {num_to_visualize} ({args.viz_percentage}%)\n')
        f.write('\n' + '=' * 50 + '\n')
        for key, value in global_metrics.items():
            f.write(f'{key}: {value:.4f}\n')

    print(f'\n✓ Results saved to: {output_dir}')
    print(f'  - Per-image metrics: {output_dir / "per_image_metrics.csv"}')
    print(f'  - Global metrics: {output_dir / "global_metrics.txt"}')
    print(f'  - Predictions: {pred_dir}')
    print(f'  - Visualizations: {pred_viz_dir}')
    print(f'  - Comparisons: {comparison_dir}')


if __name__ == '__main__':
    main()