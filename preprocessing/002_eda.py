import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import yaml
from config import *

def load_class_mapping(yaml_path):
    """Load class mapping from YAML file."""
    with open(yaml_path, 'r') as f:
        class_mapping = yaml.safe_load(f)

    # Invert mapping: value -> name
    id_to_name = {v: k for k, v in class_mapping.items()}
    return id_to_name


def analyze_defect_sizes(masks_dir, class_mapping_yaml):
    """Analyze maximum defect sizes."""

    # Load class names
    id_to_name = load_class_mapping(class_mapping_yaml)
    print(f"Loaded class mapping: {id_to_name}")

    max_widths = []
    max_heights = []

    # Get all class IDs except background (0)
    class_ids = [k for k in id_to_name.keys() if k != 0]
    defects_per_class = {class_id: [] for class_id in class_ids}

    mask_files = list(Path(masks_dir).glob("*.png"))
    total_masks = len(mask_files)

    for mask_path in tqdm(mask_files, total=total_masks, desc="Analyzing masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # For each defect class (excluding background)
        for class_id in class_ids:
            class_mask = (mask == class_id).astype(np.uint8)

            if class_mask.sum() == 0:
                continue

            # Find connected components
            num_labels, labels = cv2.connectedComponents(class_mask)

            for label_id in range(1, num_labels):
                component = (labels == label_id).astype(np.uint8)

                # Find bounding box
                coords = np.where(component > 0)
                if len(coords[0]) == 0:
                    continue

                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()

                width = x_max - x_min
                height = y_max - y_min
                diagonal = np.sqrt(width ** 2 + height ** 2)

                max_widths.append(width)
                max_heights.append(height)
                defects_per_class[class_id].append({
                    'width': width,
                    'height': height,
                    'diagonal': diagonal,
                    'area': component.sum()
                })

    # Print global statistics
    print(f"\n{'=' * 60}")
    print("GLOBAL STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total masks analyzed: {total_masks}")
    print(f"Total defects found: {len(max_widths)}")

    if len(max_widths) == 0:
        print("No defects found!")
        return None

    print(f"\nWidth statistics:")
    print(f"  Max width: {np.max(max_widths):.0f} px")
    print(f"  95th percentile: {np.percentile(max_widths, 95):.0f} px")
    print(f"  Mean width: {np.mean(max_widths):.0f} px")
    print(f"\nHeight statistics:")
    print(f"  Max height: {np.max(max_heights):.0f} px")
    print(f"  95th percentile: {np.percentile(max_heights, 95):.0f} px")
    print(f"  Mean height: {np.mean(max_heights):.0f} px")

    all_diagonals = [np.sqrt(w ** 2 + h ** 2) for w, h in zip(max_widths, max_heights)]
    print(f"\nDiagonal (max dimension):")
    print(f"  Max diagonal: {np.max(all_diagonals):.0f} px")
    print(f"  95th percentile: {np.percentile(all_diagonals, 95):.0f} px")

    # Per-class statistics
    print(f"\n{'=' * 60}")
    print("PER-CLASS STATISTICS")
    print(f"{'=' * 60}")

    for class_id in sorted(class_ids):
        class_name = id_to_name.get(class_id, f"Class {class_id}")

        if len(defects_per_class[class_id]) == 0:
            print(f"\n{class_name} (ID={class_id}): NO DATA")
            continue

        widths = [d['width'] for d in defects_per_class[class_id]]
        heights = [d['height'] for d in defects_per_class[class_id]]
        diagonals = [d['diagonal'] for d in defects_per_class[class_id]]
        areas = [d['area'] for d in defects_per_class[class_id]]

        print(f"\n{class_name} (ID={class_id}):")
        print(f"  Count: {len(defects_per_class[class_id])}")
        print(
            f"  Width  - Max: {np.max(widths):.0f}, 95%: {np.percentile(widths, 95):.0f}, Mean: {np.mean(widths):.0f}")
        print(
            f"  Height - Max: {np.max(heights):.0f}, 95%: {np.percentile(heights, 95):.0f}, Mean: {np.mean(heights):.0f}")
        print(f"  Diagonal - Max: {np.max(diagonals):.0f}, 95%: {np.percentile(diagonals, 95):.0f}")
        print(f"  Area - Mean: {np.mean(areas):.0f} px²")

    # Crop size recommendation
    print(f"\n{'=' * 60}")
    print("CROP SIZE RECOMMENDATION")
    print(f"{'=' * 60}")

    max_diagonal = np.max(all_diagonals)
    p95_diagonal = np.percentile(all_diagonals, 95)

    if p95_diagonal < 400:
        recommendation = "512x512"
        reason = "95% of defects < 400px"
    elif p95_diagonal < 600:
        recommendation = "768x768"
        reason = "95% of defects < 600px"
    elif p95_diagonal < 900:
        recommendation = "1024x1024"
        reason = "95% of defects < 900px"
    else:
        recommendation = "1024x1024 + SAHI"
        reason = f"Some very large defects (max {max_diagonal:.0f}px)"

    print(f"Recommended crop size: {recommendation}")
    print(f"Reason: {reason}")
    print(f"\nNote: With {recommendation.split()[0]} crop:")
    pct_covered = (np.array(all_diagonals) < float(recommendation.split('x')[0])).mean() * 100
    print(f"  - {pct_covered:.1f}% of defects fully contained")
    print(f"  - {100 - pct_covered:.1f}% might be cut at borders")
    print(f"{'=' * 60}\n")

    return {
        'max_widths': max_widths,
        'max_heights': max_heights,
        'defects_per_class': defects_per_class,
        'recommendation': recommendation,
        'class_names': id_to_name
    }


# Usage:
if __name__ == "__main__":
    # Path to class mapping YAML
    class_mapping_yaml = "class_mapping.yaml"  # adjust path as needed

    # Analyze training set
    results = analyze_defect_sizes(
        masks_dir=data_masks_path,
        class_mapping_yaml=class_mapping_yaml
    )

    if results:
        # Save results as YAML
        with open("defect_size_analysis.yaml", "w") as f:
            yaml.dump({
                'max_width': float(np.max(results['max_widths'])),
                'max_height': float(np.max(results['max_heights'])),
                'recommendation': results['recommendation'],
                'class_names': results['class_names']
            }, f, default_flow_style=False)
        print(f"Results saved to defect_size_analysis.yaml")