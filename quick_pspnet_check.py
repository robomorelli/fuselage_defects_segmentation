#!/usr/bin/env python3
"""
Quick diagnosis script per PSPNet che predice solo classe 0
Esegui: python quick_pspnet_check.py --model path/to/model.pth
"""

import torch
import argparse
from pathlib import Path


def quick_check(checkpoint_path):
    """Quick check del checkpoint"""

    print("\n" + "=" * 70)
    print(" PSPNET QUICK DIAGNOSIS")
    print("=" * 70)

    if not Path(checkpoint_path).exists():
        print(f"\n❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        return

    print(f"\n✓ Checkpoint loaded: {checkpoint_path}")

    # Check 1: Training info
    print(f"\n{'=' * 70}")
    print("CHECK 1: TRAINING INFO")
    print(f"{'=' * 70}")

    epoch = checkpoint.get('epoch', 'N/A')
    best_miou = checkpoint.get('best_miou', 'N/A')

    print(f"Epoch: {epoch}")
    print(f"Best mIoU: {best_miou}")

    if epoch == 'N/A' or epoch == 0:
        print("\n❌ WARNING: Epoch = 0 or N/A")
        print("   → This might be an INITIAL checkpoint, not trained!")
    elif isinstance(epoch, int) and epoch > 0:
        print(f"\n✓ Model trained for {epoch} epochs")

    if best_miou == 'N/A' or best_miou == 0 or best_miou < 0.01:
        print(f"\n❌ WARNING: Best mIoU = {best_miou}")
        print("   → Model may not be trained properly!")
    elif isinstance(best_miou, (int, float)) and best_miou > 0.1:
        print(f"\n✓ Best mIoU looks reasonable: {best_miou:.4f}")

    # Check 2: Config
    print(f"\n{'=' * 70}")
    print("CHECK 2: MODEL CONFIG")
    print(f"{'=' * 70}")

    if 'cfg' not in checkpoint:
        print("\n❌ ERROR: No 'cfg' in checkpoint!")
        return

    cfg = checkpoint['cfg']

    try:
        num_classes = cfg.model.num_classes
        architecture = cfg.model.architecture
        backbone = cfg.model.backbone
        normalize = cfg.dataset.get('normalize_imagenet', 0)

        print(f"Architecture: {architecture}")
        print(f"Backbone: {backbone}")
        print(f"Num classes: {num_classes}")
        print(f"Normalize: {normalize}")

        if num_classes != 4:
            print(f"\n⚠️ WARNING: num_classes = {num_classes} (expected 4)")
        else:
            print(f"\n✓ Num classes correct: 4")

    except Exception as e:
        print(f"\n❌ ERROR reading config: {e}")
        return

    # Check 3: State dict
    print(f"\n{'=' * 70}")
    print("CHECK 3: MODEL WEIGHTS")
    print(f"{'=' * 70}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    total_params = len(state_dict)
    print(f"Total parameters: {total_params}")

    # Check decode_head
    decode_head_keys = [k for k in state_dict.keys() if 'decode_head' in k]
    print(f"Decode head parameters: {len(decode_head_keys)}")

    if len(decode_head_keys) == 0:
        print("\n❌ ERROR: No decode_head found!")
        print("   → Model may not be loaded correctly")

    # Check final conv
    conv_seg_weight = None
    for key in state_dict.keys():
        if 'conv_seg.weight' in key or 'classifier.weight' in key:
            conv_seg_weight = state_dict[key]
            print(f"\nFinal conv layer: {key}")
            print(f"  Shape: {conv_seg_weight.shape}")

            if conv_seg_weight.shape[0] != num_classes:
                print(f"  ❌ ERROR: Output channels = {conv_seg_weight.shape[0]}, expected {num_classes}")
            else:
                print(f"  ✓ Output channels correct: {num_classes}")

            # Check if weights are trained
            weight_mean = conv_seg_weight.abs().mean().item()
            weight_std = conv_seg_weight.std().item()

            print(f"  Weights mean (abs): {weight_mean:.6f}")
            print(f"  Weights std: {weight_std:.6f}")

            if weight_mean < 0.001:
                print(f"\n  ❌ CRITICAL: Weights very small ({weight_mean:.6f})")
                print("     → Model is likely NOT TRAINED!")
            elif weight_mean > 0.01:
                print(f"\n  ✓ Weights look trained ({weight_mean:.6f})")
            else:
                print(f"\n  ⚠️ Weights suspicious ({weight_mean:.6f})")

            break

    if conv_seg_weight is None:
        print("\n❌ ERROR: Could not find final conv layer!")

    # Final diagnosis
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS")
    print(f"{'=' * 70}")

    issues = []

    if epoch == 'N/A' or epoch == 0:
        issues.append("Epoch = 0 → Model NOT trained")

    if best_miou == 'N/A' or best_miou == 0:
        issues.append("Best mIoU = 0 → Model NOT trained")

    if conv_seg_weight is not None and conv_seg_weight.abs().mean().item() < 0.001:
        issues.append("Weights very small → Model NOT trained")

    if len(decode_head_keys) == 0:
        issues.append("No decode_head → Model corrupted")

    if issues:
        print("\n❌ PROBLEMS FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nLIKELY CAUSE: You are using an INITIAL/UNTRAINED checkpoint!")
        print("\nSOLUTIONS:")
        print("  1. Check your experiments folder for the FINAL checkpoint")
        print("  2. Look for checkpoint with highest epoch number")
        print("  3. Verify training completed successfully (check logs)")
        print("  4. Use checkpoint from end of training, not beginning")
    else:
        print("\n✓ Checkpoint looks OK")
        print("\nIf model still predicts only class 0, check:")
        print("  1. Preprocessing matches training (normalize flag)")
        print("  2. Model in eval mode: model.eval()")
        print("  3. Using torch.no_grad() during inference")

    print("\n" + "=" * 70)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./fuselage_segmentation_hpo_pspnet/an77vi1e/pspnet/resnet101/fold_1/pspnet_hpo_w_1_1_1.05_09_02_2026_14_48_07/azure-sweep-1/pspnet_resnet101_fold1.pth", help='Path to checkpoint')
    args = parser.parse_args()

    quick_check(args.model)