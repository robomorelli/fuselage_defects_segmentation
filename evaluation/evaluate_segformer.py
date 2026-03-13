"""
Updated SegFormer Evaluation Script
- Statistiche predizioni accumulate
- Visualizzazioni salvate in checkpoint folder
- Debug prints
- Omogeneo con evaluation MMSeg
"""

import os
import shutil
import argparse
import numpy as np
import sys

sys.path.append('..')
from dataset.segmentation import KFoldDataframeMulticlassProcessor
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import *
from evaluation.eval_utils import compute_iou_multiclass, compute_metrics_multiclass
from tqdm import tqdm
import cv2
import yaml
import torch.nn as nn
from transformers import (SegformerForSemanticSegmentation, SegformerImageProcessor)

# Visualizations
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def save_evaluation_visualizations(images_list, preds_list, gts_list, filenames_list,
                                   save_dir, num_samples=10):
    """
    Salva visualizzazioni delle predizioni in evaluation
    Simile a quella del training ma per evaluation
    """

    save_dir = Path(save_dir) / 'eval_predictions'
    save_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, len(images_list))
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])

    for i in range(num_samples):
        img = images_list[i]
        pred_mask = preds_list[i]
        gt_mask = gts_list[i]
        filename = filenames_list[i]

        # img è già BGR da cv2.imread
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        unique_pred = np.unique(pred_mask)
        unique_gt = np.unique(gt_mask)

        # Crea figura
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img)
        axes[0].set_title(f'Image: {filename}', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=3)
        axes[1].set_title(f'GT: {unique_gt.tolist()}', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=3)
        axes[2].set_title(f'Pred: {unique_pred.tolist()}', fontsize=12)
        axes[2].axis('off')

        # Percentuali per classe
        for ax, mask, title in [(axes[1], gt_mask, 'GT'), (axes[2], pred_mask, 'Pred')]:
            text = []
            for cls in range(4):
                pct = (mask == cls).sum() / mask.size * 100
                if pct > 0.01:
                    text.append(f'C{cls}:{pct:.1f}%')
            ax.text(0.5, -0.05, ' '.join(text),
                    transform=ax.transAxes, ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_dir / f'eval_{i:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"\n✓ Saved {num_samples} evaluation visualizations to: {save_dir}")


def main(data_path, model_path, ths_num=0, unique_th=0.4,
         df_path=k_fold_data_path, split='test', save_into_common_folder=False,
         save_into_model_folder=False, multi_ths=0, reduce_labels=1, ignore_index=255,
         f1_metrics=0, iou_metrics=1, from_full_to_crop=1,
         cropped=1, load_predictions=0, predictions_folder=model_results):
    if not os.path.exists(model_path):
        print('the model path is not correct')
        raise Exception

    # ========================================================================
    # Setup paths - MODIFICATO per salvare nella cartella checkpoint
    # ========================================================================
    save_path = os.path.join(Path(model_path).parent.as_posix())
    fold = os.path.basename(Path(model_path).parent.parent).split('_')[1]

    print(f"\n{'=' * 70}")
    print(f"SEGFORMER EVALUATION")
    print(f"{'=' * 70}")
    print(f"Model path: {model_path}")
    print(f"Save path: {save_path}")
    print(f"Fold: {fold}")
    print(f"Split: {split}")

    # Metrics path setup
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

    # ========================================================================
    # Load model
    # ========================================================================
    num_classes = 3
    class_names = ['mark', 'graffio', 'drill start']
    print(f'\nMulticlass for n classes: {num_classes}')

    if 'model.' in model_path:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
    else:
        checkpoint = torch.load(os.path.join(model_path, "segformer.pth"), map_location=torch.device(device))

    cfg = checkpoint['cfg']

    # Check checkpoint info
    print(f"\nCheckpoint info:")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_miou' in checkpoint:
        print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A')}")

    # Build model
    encoder_name = cfg.model.encoder_name
    print(f'\nLoading SegFormer with {encoder_name} encoder')

    with open('./preprocessing/class_mapping.yaml', 'r') as stream:
        try:
            label2id = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    label2id['bkg'] = 0
    id2label = {v: k for k, v in label2id.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        encoder_name,
        num_labels=num_classes + 1,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    print('Classes dict:', id2label)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()  # IMPORTANTE!

    print(f"\n✓ Model loaded and set to eval mode")
    print(f"  Device: {device}")
    print(f"  Training mode: {model.training}")  # Should be False

    # ========================================================================
    # Setup dataset
    # ========================================================================
    if 'cfg' in checkpoint.keys():
        cfg = checkpoint['cfg']
        normalize_imagenet = cfg.dataset.normalize_imagenet
        print(f'\nFrom cfg: normalize_imagenet = {normalize_imagenet}')
    else:
        normalize_imagenet = 0
        print(f'\nNo cfg, using normalize_imagenet = {normalize_imagenet}')

    if not cfg.opt.processor:
        processor = None
        print('No processor')
    else:
        processor = SegformerImageProcessor.from_pretrained(cfg.model.encoder_name)
        print(f'Using SegformerImageProcessor')

    transform = None
    if 'k-fold' in df_path:
        df_path = os.path.join(df_path, f"fold_{fold}", split)
    else:
        df_path = os.path.join(df_path, split)

    print(f'\nDataset:')
    print(f'  Data path: {data_path}')
    print(f'  DF path: {df_path}')
    print(f'  Normalize: {normalize_imagenet}')
    print(f'  Cropped: {cropped}')

    dataset = KFoldDataframeMulticlassProcessor(
        data_path,
        df_path=df_path,
        transform=transform,
        normalize_imagenet=normalize_imagenet,
        cropped=cropped,
        from_full_to_crop=from_full_to_crop,
        processor=processor
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f'✓ Dataset created: {len(dataset)} images\n')

    # ========================================================================
    # Setup metrics
    # ========================================================================
    if ths_num > 0 and multi_ths:
        ths = np.linspace(0.2, 0.95, ths_num)
    elif unique_th > 0:
        ths = [unique_th]
    else:
        ths = [0.5]

    if f1_metrics:
        metrics_name = ["TP", "FP", "FN"]
        global_metrics_name = ["F1", "TP", "FP", "FN", "accuracy", "precision", "recall"]
        columns = [name1 + '_' + name2 for name1 in class_names for name2 in metrics_name]
        global_columns = [name1 + '_' + name2 for name1 in class_names for name2 in global_metrics_name]
        metrics_dicts = {f"{th}": pd.DataFrame(None, columns=columns) for th in ths}
        global_metrics = pd.DataFrame(None, columns=global_columns)

    elif iou_metrics:
        columns = [f'class_{ix + 1}_iou' for ix in range(num_classes)]
        columns = columns + [f'class_{ix + 1}_accuracy' for ix in range(num_classes)]
        columns = columns + ["loss"]
        global_metrics = pd.DataFrame(None, columns=columns)

        results_dict_iou = {f'class_{ix + 1}_iou': [] for ix in range(num_classes)}
        results_dict_accuracy = {f'class_{ix + 1}_accuracy': [] for ix in range(num_classes)}

    # ========================================================================
    # NUOVO: Accumulatori statistiche predizioni
    # ========================================================================
    pred_pixels = np.zeros(num_classes + 1)  # +1 per background
    gt_pixels = np.zeros(num_classes + 1)
    total_pixels = 0

    # Per visualizzazioni
    viz_images = []
    viz_preds = []
    viz_gts = []
    viz_filenames = []

    criterion = torch.nn.CrossEntropyLoss()
    filenames = dataset.images_file_names

    # ========================================================================
    # Setup save directories (mantieni features esistenti)
    # ========================================================================
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
            if not os.path.exists(save_into_common_path_viz):
                os.makedirs(save_into_common_path_viz)

    if save_into_model_folder:
        for th in ths:
            save_into_model_path = os.path.join(save_path, f"{split}/model_results_{th}")
            save_into_model_path_viz = os.path.join(save_path, f"{split}/model_results_{th}_viz")

            if not os.path.exists(save_into_model_path):
                os.makedirs(save_into_model_path)
            if not os.path.exists(save_into_model_path_viz):
                os.makedirs(save_into_model_path_viz)

    # ========================================================================
    # EVALUATION LOOP
    # ========================================================================
    print(f"{'=' * 70}")
    print(f"STARTING EVALUATION")
    print(f"{'=' * 70}\n")

    with torch.no_grad():
        running_loss = 0.0

        for i, (im, gt_mask) in tqdm(enumerate(dataloader), total=len(dataset)):

            if gt_mask.max() > num_classes:
                for j in range(num_classes):
                    gt_mask[gt_mask == np.ceil(255 / (j + 1))] = num_classes - j

            name = filenames[i]

            if not load_predictions:
                logits = model(im.to(device)).logits

                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=tuple(im.shape[-2:]),
                    mode='bilinear',
                    align_corners=False
                )

                loss = criterion(upsampled_logits.float(), gt_mask.to(device).squeeze(1).long()).item()

                pred_mask = upsampled_logits[0].softmax(0).permute(1, 2, 0).detach().cpu().numpy()
                pred_mask = np.argmax(pred_mask, 2)

                running_loss += loss
                mean_loss = running_loss / (i + 1)

                gt_fh = dataset.images_file_names[i]
                gt_mask = cv2.imread(os.path.join(data_path, 'masks', gt_fh.replace('.', '_mask.')))
                gt_mask = np.squeeze(cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)[:, :, 0:1])

            else:
                gt_fh = dataset.images_file_names[i]
                gt_mask = cv2.imread(os.path.join(data_path, 'masks', gt_fh.replace('.', '_mask.')))
                gt_mask = np.squeeze(cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)[:, :, 0:1])

                pred = cv2.imread(os.path.join(predictions_folder, gt_fh))
                pred_mask = np.squeeze(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)[:, :, 0:1])
                mean_loss = None

            # ================================================================
            # NUOVO: Debug print primi 2 batch
            # ================================================================
            if i < 2:
                print(f"\n{'=' * 70}")
                print(f"BATCH {i} DEBUG")
                print(f"{'=' * 70}")
                print(f"Filename: {gt_fh}")
                print(f"GT unique: {np.unique(gt_mask)}")
                print(f"Pred unique: {np.unique(pred_mask)}")

                # Per-class counts
                for cls in range(num_classes + 1):
                    pred_count = (pred_mask == cls).sum()
                    gt_count = (gt_mask == cls).sum()
                    total = pred_mask.size
                    print(f"Class {cls}: pred={pred_count} ({pred_count / total * 100:.1f}%), "
                          f"gt={gt_count} ({gt_count / total * 100:.1f}%)")
                print(f"{'=' * 70}\n")
            # ================================================================

            # ================================================================
            # NUOVO: Accumula statistiche
            # ================================================================
            for cls in range(num_classes + 1):
                pred_pixels[cls] += (pred_mask == cls).sum()
                gt_pixels[cls] += (gt_mask == cls).sum()
            total_pixels += pred_mask.size
            # ================================================================

            # ================================================================
            # NUOVO: Raccogli prime 10 per visualizzazione
            # ================================================================
            if i < 10:
                img_bgr = cv2.imread(os.path.join(data_path, 'images', gt_fh))
                viz_images.append(img_bgr)
                viz_preds.append(pred_mask.copy())
                viz_gts.append(gt_mask.copy())
                viz_filenames.append(gt_fh)
            # ================================================================

            # Compute metrics (codice esistente)
            if iou_metrics:
                results, pred = compute_iou_multiclass(
                    gt_mask, pred_mask, img_name=gt_fh,
                    obj_size=args.remove_small_objs_size,
                    reduce_labels=reduce_labels, ignore_index=ignore_index
                )

                for j, k in enumerate(list(results_dict_iou.keys())):
                    results_dict_iou[f'class_{j + 1}_iou'].append(results['per_category_iou'][j])
                for j, k in enumerate(list(results_dict_accuracy.keys())):
                    results_dict_accuracy[f'class_{j + 1}_accuracy'].append(results['per_category_accuracy'][j])

            else:
                for th in ths:
                    metrics = metrics_dicts[str(th)]
                    metrics_dicts[str(th)], pred = compute_metrics_multiclass(
                        gt_mask, pred_mask, metrics=metrics, img_name=gt_fh, th=th,
                        n_classes=num_classes, obj_size=args.remove_small_objs_size,
                        return_pred_th=True, id_labels_dict=id2label
                    )

            # Save predictions (codice esistente)
            if save_into_common_folder:
                # ... codice esistente ...
                pass

            if save_into_model_folder:
                cv2.imwrite(os.path.join(save_into_model_path, f"{name}"), np.squeeze(pred))
                pred_viz = (np.array(pred) / num_classes) * 255
                cv2.imwrite(os.path.join(save_into_model_path_viz, f"{name}"), np.squeeze(pred_viz))

    # ========================================================================
    # NUOVO: Stampa statistiche finali
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"EVALUATION PREDICTION STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total pixels: {total_pixels:,}\n")

    for cls in range(num_classes + 1):
        pred_pct = (pred_pixels[cls] / total_pixels) * 100
        gt_pct = (gt_pixels[cls] / total_pixels) * 100
        print(f"Class {cls}:")
        print(f"  Predicted: {int(pred_pixels[cls]):,} pixels ({pred_pct:.2f}%)")
        print(f"  GT:        {int(gt_pixels[cls]):,} pixels ({gt_pct:.2f}%)")

    print(f"\n{'=' * 70}")

    # Diagnosi
    if pred_pixels[0] == total_pixels:
        print("\n❌ CRITICAL: Model predicts ONLY class 0 (100%)!")
    elif pred_pixels[0] / total_pixels > 0.95:
        print(f"\n⚠️ WARNING: Model predicts mostly class 0 ({pred_pixels[0] / total_pixels * 100:.1f}%)")
    else:
        print("\n✓ Model predicts multiple classes")

    print(f"\n{'=' * 70}\n")

    # ========================================================================
    # NUOVO: Salva visualizzazioni
    # ========================================================================
    save_evaluation_visualizations(
        viz_images, viz_preds, viz_gts, viz_filenames,
        save_dir=save_path,
        num_samples=10
    )

    # ========================================================================
    # Save metrics (codice esistente)
    # ========================================================================
    if f1_metrics:
        outname = os.path.join(save_path, f'{split_suffix}_global_metrics.csv')
        global_metrics.to_csv(outname, index=True, index_label='Threshold')
    else:
        global_values = []
        for k in list(results_dict_iou.keys()):
            results_dict_iou[k] = [x for x in results_dict_iou[k] if not (np.isnan(x) or x == 0)]
            global_values.append(np.mean(results_dict_iou[k]))
        for k in list(results_dict_accuracy.keys()):
            results_dict_accuracy[k] = [x for x in results_dict_accuracy[k] if not (np.isnan(x) or x == 0)]
            global_values.append(np.mean(results_dict_accuracy[k]))

        global_metrics.iloc[0, :] = global_values

        outname = os.path.join(save_path, f'{split_suffix}_global_metrics.csv')
        global_metrics.to_csv(outname, index=False)
        print(f"\n✓ Saved metrics to: {outname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SegFormer Evaluation with Statistics")
    parser.add_argument("--ths_num", default=1, type=int)
    parser.add_argument("--multi_ths", default=1, type=int)
    parser.add_argument("--unique_th", default=0.5, type=float)
    parser.add_argument("--model_path", default="./segformer_hyp_opt_dv_3_classes_test/r9cpb1ol/segformer/nvidia/mit-b3/fold_1/1_fold_w_1_1_1.05_09_02_2026_17_40_02/eager-sweep-1")
    parser.add_argument("--load_predictions", default=0, type=int)
    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument("--predictions_folder", default=model_results)
    parser.add_argument("--data_path", default=cropped_tot_bkg_data_path)
    parser.add_argument("--df_path", default=k_fold_data_path)
    parser.add_argument("--cropped", default=1, type=int)
    parser.add_argument("--from_full_to_crop", default=1, type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--remove_small_objs_size", default=100, type=int)
    parser.add_argument("--save_into_common_folder", default=0, type=int)
    parser.add_argument("--save_into_model_folder", default=1, type=int)
    parser.add_argument("--reduce_labels", default=0, type=int)
    parser.add_argument("--ignore_index", default=255, type=int)
    parser.add_argument("--f1_metrics", default=1, type=int)
    parser.add_argument("--iou_metrics", default=0, type=int)

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        ths_num=args.ths_num,
        unique_th=args.unique_th,
        df_path=args.df_path,
        split=args.split,
        save_into_common_folder=args.save_into_common_folder,
        save_into_model_folder=args.save_into_model_folder,
        multi_ths=args.multi_ths,
        reduce_labels=args.reduce_labels,
        ignore_index=args.ignore_index,
        f1_metrics=args.f1_metrics,
        iou_metrics=args.iou_metrics,
        from_full_to_crop=args.from_full_to_crop,
        cropped=args.cropped,
        load_predictions=args.load_predictions,
        predictions_folder=args.predictions_folder
    )