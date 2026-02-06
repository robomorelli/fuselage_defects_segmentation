import os
import shutil
import argparse
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
import sys

sys.path.append('..')
from dataset.segmentation import KFoldDataframeMulticlassProcessor_v2
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import *
from evaluation.utils import compute_metrics_multiclass, compute_iou_multiclass
from tqdm import tqdm
import cv2
import yaml
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import random

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_color_map(num_classes):
    """Crea una colormap per le classi"""
    colors = sns.color_palette("husl", num_classes + 1)
    return ListedColormap(colors)


def visualize_comparison(image, gt_mask, pred_mask, name, save_path, id2label, colormap):
    """
    Crea una visualizzazione a 4 pannelli:
    - Immagine originale
    - Ground Truth
    - Predizione
    - Overlay (GT in rosso, Pred in verde, overlap in giallo)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Immagine originale
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalizza se necessario
        if img_np.min() < 0:
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = image

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Immagine Originale', fontsize=14)
    axes[0, 0].axis('off')

    # Ground Truth
    gt_colored = axes[0, 1].imshow(gt_mask, cmap=colormap, vmin=0, vmax=len(id2label) - 1)
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')

    # Predizione
    pred_colored = axes[1, 0].imshow(pred_mask, cmap=colormap, vmin=0, vmax=len(id2label) - 1)
    axes[1, 0].set_title('Predizione', fontsize=14)
    axes[1, 0].axis('off')

    # Overlay per errori
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    # Rosso: GT ma non predetto (False Negative)
    overlay[(gt_mask > 0) & (pred_mask == 0)] = [255, 0, 0]
    # Verde: Predetto ma non GT (False Positive)
    overlay[(gt_mask == 0) & (pred_mask > 0)] = [0, 255, 0]
    # Giallo: Correttamente predetto (True Positive)
    overlay[(gt_mask > 0) & (pred_mask > 0) & (gt_mask == pred_mask)] = [255, 255, 0]
    # Bianco: classi diverse predette
    overlay[(gt_mask > 0) & (pred_mask > 0) & (gt_mask != pred_mask)] = [255, 255, 255]

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay Errori\n(Rosso=FN, Verde=FP, Giallo=TP, Bianco=Wrong Class)',
                         fontsize=12)
    axes[1, 1].axis('off')

    # Aggiungi colorbar con le etichette delle classi
    cbar = plt.colorbar(pred_colored, ax=axes[1, 0], orientation='horizontal',
                        pad=0.05, fraction=0.046)
    cbar.set_ticks(range(len(id2label)))
    cbar.set_ticklabels([id2label[i] for i in range(len(id2label))], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def compute_per_image_metrics(gt_mask, pred_mask, num_classes):
    """Calcola metriche per singola immagine"""
    metrics = {}

    # IoU per classe
    ious = []
    for cls in range(num_classes + 1):
        if cls == 0:  # skip background se vuoi
            continue
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


def main(data_path, model_path, model_type='deeplabv3', exp_name=None, viz_percentage=100):
    num_classes = len(os.listdir(full_size_masks_classes_path))
    print(f'Multiclass segmentation for {num_classes} classes')

    if not os.path.exists(model_path):
        print('Model path does not exist')
        raise Exception

    # Determina la directory del modello
    if os.path.isfile(model_path):
        model_dir = Path(model_path).parent
    else:
        model_dir = Path(model_path)

    # Crea cartella risultati allo stesso livello del checkpoint
    if exp_name is not None:
        save_path = os.path.join(model_dir, exp_name)
    else:
        save_path = os.path.join(model_dir, f"evaluation_{os.path.basename(Path(data_path).parent)}")

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print(f'Results will be saved in: {save_path}')

    # Carica checkpoint
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(device))
    else:
        checkpoint = torch.load(os.path.join(model_path, "model.pth"),
                                map_location=torch.device(device))

    cfg = checkpoint.get('cfg', None)

    # Carica mapping classi
    with open('../preprocessing/class_mapping.yaml', 'r') as stream:
        label2id = yaml.safe_load(stream)
    label2id['bkg'] = 0
    id2label = {v: k for k, v in label2id.items()}

    # Inizializza modello
    if model_type == 'deeplabv3':
        print('Loading DeepLabV3 model...')
        model = deeplabv3_resnet101(pretrained=False)
        model.classifier = DeepLabHead(2048, num_classes=num_classes + 1)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif model_type == 'pspnet':
        print('Loading PSPNet model...')
        # Assumendo che tu abbia una funzione/classe per PSPNet
        from models.pspnet import PSPNet  # adatta al tuo import
        model = PSPNet(num_classes=num_classes + 1)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    model.eval()

    # Normalize ImageNet
    normalize_imagenet = cfg.dataset.normalize_imagenet if cfg else 0
    print(f'Normalize ImageNet: {normalize_imagenet}')

    # Dataset
    dataset = KFoldDataframeMulticlassProcessor_v2(
        data_path,
        df_path=None,
        from_folder=True,
        transform=None,
        normalize_imagenet=normalize_imagenet,
        from_full_to_crop=True,
        processor=None
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    filenames = dataset.images_file_names

    # Crea directories
    save_pred_path = os.path.join(save_path, "predictions")
    save_pred_viz_path = os.path.join(save_path, "predictions_viz")
    save_comparison_path = os.path.join(save_path, "comparisons")

    os.makedirs(save_pred_path, exist_ok=True)
    os.makedirs(save_pred_viz_path, exist_ok=True)
    os.makedirs(save_comparison_path, exist_ok=True)

    # Colormap
    colormap = create_color_map(num_classes)

    # Determina quali immagini salvare per le visualizzazioni
    total_images = len(dataset)
    num_to_visualize = int(total_images * viz_percentage / 100)

    if num_to_visualize < total_images:
        # Seleziona casualmente quali immagini visualizzare
        viz_indices = set(random.sample(range(total_images), num_to_visualize))
        print(f'Saving visualizations for {num_to_visualize}/{total_images} images ({viz_percentage}%)')
    else:
        viz_indices = set(range(total_images))
        print(f'Saving visualizations for all {total_images} images')

    # Metriche globali
    all_metrics = []
    all_ious = []

    print('\nStarting evaluation...')
    with torch.no_grad():
        for i, (im, gt_mask) in tqdm(enumerate(dataloader), total=len(dataset)):

            # Fix ground truth se necessario
            if gt_mask.max() > num_classes:
                for cls in range(num_classes):
                    gt_mask[gt_mask == np.ceil(255 / (cls + 1))] = num_classes - cls

            name = os.path.splitext(filenames[i])[0]

            # Inferenza
            if model_type == 'deeplabv3':
                output = model(im.to(device))['out']
            else:  # pspnet
                output = model(im.to(device))

            # Upsample
            upsampled_logits = nn.functional.interpolate(
                output,
                size=tuple(im.shape[-2:]),
                mode='bilinear',
                align_corners=False
            )

            # Predizione
            pred_mask = upsampled_logits[0].softmax(0).argmax(0).cpu().numpy()
            gt_mask_np = gt_mask.squeeze().cpu().numpy()

            # Calcola metriche
            metrics = compute_per_image_metrics(gt_mask_np, pred_mask, num_classes)
            all_metrics.append({
                'filename': filenames[i],
                'mIoU': metrics['mIoU'],
                'pixel_acc': metrics['pixel_acc']
            })
            all_ious.extend(metrics['per_class_IoU'])

            # Salva predizioni (sempre)
            cv2.imwrite(os.path.join(save_pred_path, f"{name}.png"), pred_mask.astype(np.uint8))

            # Salva visualizzazione colorata (sempre)
            pred_viz = (pred_mask / num_classes * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_pred_viz_path, f"{name}.png"), pred_viz)

            # Crea visualizzazione comparativa (solo per % selezionata)
            if i in viz_indices:
                visualize_comparison(
                    im.squeeze(),
                    gt_mask_np,
                    pred_mask,
                    name,
                    save_comparison_path,
                    id2label,
                    colormap
                )

    # Salva metriche
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(save_path, 'per_image_metrics.csv'), index=False)

    # Metriche globali
    global_metrics = {
        'mean_mIoU': metrics_df['mIoU'].mean(),
        'std_mIoU': metrics_df['mIoU'].std(),
        'mean_pixel_acc': metrics_df['pixel_acc'].mean(),
        'std_pixel_acc': metrics_df['pixel_acc'].std(),
        'overall_mIoU': np.mean(all_ious)
    }

    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    for key, value in global_metrics.items():
        print(f'{key}: {value:.4f}')
    print('=' * 50)

    # Salva metriche globali
    with open(os.path.join(save_path, 'global_metrics.txt'), 'w') as f:
        f.write(f'Evaluation on: {data_path}\n')
        f.write(f'Model: {model_path}\n')
        f.write(f'Model type: {model_type}\n')
        f.write(f'Number of classes: {num_classes}\n')
        f.write(f'Total images: {total_images}\n')
        f.write(f'Visualizations saved: {num_to_visualize} ({viz_percentage}%)\n')
        f.write('\n' + '=' * 50 + '\n')
        for key, value in global_metrics.items():
            f.write(f'{key}: {value:.4f}\n')

    print(f'\nResults saved in: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PSPNet/DeepLab segmentation models")
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint or directory")
    parser.add_argument("--data_path", default=cropped_may_data_path,
                        help="Path to evaluation data")
    parser.add_argument("--model_type", default='deeplabv3',
                        choices=['deeplabv3', 'pspnet'],
                        help="Type of model to evaluate")
    parser.add_argument("--exp_name", default=None,
                        help="Experiment name for output folder (default: evaluation_<data_name>)")
    parser.add_argument("--viz_percentage", type=float, default=100.0,
                        help="Percentage of images to save with comparison visualizations (0-100, default: 100)")

    args = parser.parse_args()

    # Valida viz_percentage
    if not 0 <= args.viz_percentage <= 100:
        parser.error("viz_percentage must be between 0 and 100")

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        model_type=args.model_type,
        exp_name=args.exp_name,
        viz_percentage=args.viz_percentage
    )