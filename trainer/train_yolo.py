"""
YOLOv8-seg training script for defect detection
Equivalent to train.py for Segformer but adapted for YOLO
"""

import os
import torch
import torch.optim
from datetime import datetime
import wandb
from box import Box
import subprocess
from ultralytics import YOLO
from pathlib import Path
import yaml


def get_gpu_memory():
    """
    Returns the memory usage of all available GPUs on the node.
    Uses `nvidia-smi` to get the GPU memory stats.

    Returns:
        list: List of tuples (gpu_index, memory_used, memory_free, memory_total)
    """
    gpu_info = []
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,nounits,noheader'])
        result = result.decode('utf-8').strip().split('\n')
        for line in result:
            parts = line.split(', ')
            gpu_index = int(parts[0])  # GPU index
            memory_free = int(parts[1])  # Free memory (in MB)
            memory_total = int(parts[2])  # Total memory (in MB)
            memory_used = memory_total - memory_free
            gpu_info.append(
                (gpu_index, memory_used, memory_free, memory_total))
    except Exception as e:
        print(f"Error retrieving GPU memory information: {e}")
    return gpu_info


def select_gpus(n=2):
    """
    Selects the N GPUs with the lowest memory usage (highest free memory).

    Args:
        n (int): Number of GPUs to select

    Returns:
        list: List of selected GPU indices
    """
    gpu_info = get_gpu_memory()
    if not gpu_info:
        raise ValueError("No GPU information found.")

    # Sort GPUs by free memory (descending)
    sorted_gpus = sorted(gpu_info, key=lambda x: x[2], reverse=True)

    # Select the top N GPUs
    print(f"Available GPUs sorted by free memory: {sorted_gpus}")
    selected_gpus = [gpu[0] for gpu in sorted_gpus[:n]]

    print(f"Selected GPUs: {selected_gpus}")
    return selected_gpus


def prepare_data_yaml(cfg, fold):
    """
    Prepares the data.yaml file for the current fold.
    Creates YOLO-format dataset configuration.

    Args:
        cfg: Configuration Box object
        fold: Fold number for cross-validation

    Returns:
        str: Path to the created data.yaml file
    """
    data_path = cfg.dataset.data_path

    # Build path for this fold
    fold_data_path = os.path.join(data_path, f'fold_{fold}')

    # Verify fold directory exists
    if not os.path.exists(fold_data_path):
        raise ValueError(f"Fold directory not found: {fold_data_path}")

    # Define class names (modify these for your actual class names)
    class_names = ['scratch', 'mark', 'drill_hole']  # For your 3-class case

    # Create data.yaml for this fold
    data_yaml = {
        'path': os.path.abspath(fold_data_path),  # Absolute root directory
        'train': 'images/train',  # Train images (relative to 'path')
        'val': 'images/val',      # Val images (relative to 'path')
        'nc': cfg.model.num_classes,  # Number of classes
        'names': class_names[:cfg.model.num_classes]  # Class names
    }

    # Save data.yaml
    data_yaml_path = os.path.join(fold_data_path, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Data YAML created at: {data_yaml_path}")
    return data_yaml_path


def train():
    """
    Main training function called by wandb agent for each sweeps run.
    This is the equivalent of the Segformer train() function but for YOLO.
    """
    # Initialize wandb for this run
    group = os.getenv("WANDB_GROUP", None)
    if group is not None:
        run = wandb.init(group=group)
    else:
        run = wandb.init()

    # ========== EXTRACT AND CAST CONFIG VALUES ==========
    # Explicitly cast config values to appropriate types (same as Segformer)
    cfg = Box(wandb.config.get('cfg'))

    # Set wandb to offline mode if specified
    if not cfg.wandb.online:
        os.environ["WANDB_MODE"] = "offline"

    # Extract sweeps parameters and cast to correct types
    cfg.dataset.fold = int(wandb.config.get('fold', '1'))
    cfg.opt.weights = list(wandb.config.get('classes_weights'))
    cfg.opt.lr = float(wandb.config.get('lr', '0.0001'))
    cfg.opt.es_patience = int(wandb.config.get('es_patience', '8'))

    # For YOLO, model_size determines the model variant (n, s, m, l, x)
    model_size = str(wandb.config.get('model_size', 'm'))
    cfg.model.encoder_name = f'yolov8{model_size}-seg'

    # Define wandb metric for optimization
    summary_dict = {'maximize': "max", "minimize": 'min'}
    wandb.define_metric(cfg.opt.metric_name, summary=summary_dict[cfg.opt.metric_goal] + ',last')

    # ========== SETUP DIRECTORIES ==========
    # Generate unique timestamp for this run
    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # Get sweeps information
    sweep_id = run.sweep_id
    api = wandb.Api()
    entity = run.entity
    project = run.project
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    sweep_name = sweep.name

    fold = int(wandb.config.get('fold', '1'))

    # Create model directory structure (same as Segformer)
    model_dir = os.path.join(
        wandb.config.get('project_name'),
        sweep_name,
        cfg.model.name,
        cfg.model.encoder_name,
        f"fold_{fold}",
        cfg.model.exp_name + f'_w_{cfg.opt.weights[0]}_{cfg.opt.weights[1]}_{cfg.opt.weights[2]}' + "_" + now,
        run.name
    )

    os.makedirs(model_dir, exist_ok=True)
    cfg.model.checkpoint = os.path.join(model_dir, cfg.model.name + '.pt')

    # ========== MODEL INITIALIZATION ==========
    # Load checkpoint or pretrained model
    if cfg.model.checkpoint and os.path.exists(cfg.model.checkpoint):
        print(f"Loading checkpoint: {cfg.model.checkpoint}")
        model = YOLO(cfg.model.checkpoint)
    else:
        # Use pretrained model by default
        model_path = f'yolov8{model_size}-seg.pt'
        print(f"Loading pretrained model: {model_path}")
        model = YOLO(model_path)

    # ========== GPU SELECTION ==========
    # Check for environment variable for device selection
    device_env = os.getenv("GPU_ID")
    if device_env:
        # Use specific GPU from environment
        device_id = int(device_env)
        device = device_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(f"Using GPU from GPU_ID env variable: {device_id}")
    else:
        # Select N GPUs with least memory usage
        n_gpus = int(wandb.config.get('n_gpus', '1'))
        print(f'Requesting {n_gpus} GPU(s)')

        if n_gpus > 1:
            selected_gpus = select_gpus(n=n_gpus)
            device_ids = list(range(0, n_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
            device = device_ids  # YOLO handles multi-GPU automatically
            print(f"Using GPUs {selected_gpus} for multi-GPU training")
        else:
            selected_gpus = select_gpus(n=1)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpus[0])
            device = 0  # First (and only) visible GPU
            print(f"Using single GPU: {selected_gpus[0]}")

    # ========== CPU THREADING CONFIGURATION ==========
    ncpus = int(wandb.config.get('ncpus', '0'))
    if ncpus == 0:
        n_gpus = int(wandb.config.get('n_gpus', '1'))
        ncpus = n_gpus * 12  # Default: 12 CPUs per GPU

    torch.set_num_threads(ncpus)
    num_workers = ncpus

    # Adjust batch size based on number of GPUs (same as Segformer)
    batch_size = int(wandb.config.get('batch', '2')) * n_gpus

    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["MKL_NUM_THREADS"] = str(ncpus)

    print(f"CPU configuration: {ncpus} threads, {num_workers} workers")
    print(f"Batch size: {batch_size} (base: {wandb.config.get('batch', '2')} × {n_gpus} GPUs)")

    # ========== PREPARE DATA CONFIGURATION ==========
    data_yaml_path = prepare_data_yaml(cfg, fold)

    # ========== TRAINING ARGUMENTS ==========
    train_args = {
        # Data
        'data': data_yaml_path,

        # Training duration
        'epochs': int(wandb.config.get('epochs', '200')),
        'patience': cfg.opt.es_patience,  # Early stopping patience

        # Batch and workers
        'batch': batch_size,
        'workers': num_workers,

        # Image size
        'imgsz': int(wandb.config.get('imgsz', '640')),

        # Optimizer settings
        'optimizer': wandb.config.get('optimizer', 'Adam'),
        'lr0': cfg.opt.lr,
        'lrf': float(wandb.config.get('lrf', '0.01')),
        'momentum': float(wandb.config.get('momentum', '0.937')),
        'weight_decay': float(wandb.config.get('weight_decay', '0.0005')),

        # Warmup settings
        'warmup_epochs': float(wandb.config.get('warmup_epochs', '3.0')),
        'warmup_momentum': float(wandb.config.get('warmup_momentum', '0.8')),
        'warmup_bias_lr': float(wandb.config.get('warmup_bias_lr', '0.1')),

        # Augmentation (only if augmentation: 1 in cfg)
        'augment': cfg.dataset.augmentation == 1,
        'hsv_h': float(wandb.config.get('hsv_h', '0.015')) if cfg.dataset.augmentation else 0.0,
        'hsv_s': float(wandb.config.get('hsv_s', '0.7')) if cfg.dataset.augmentation else 0.0,
        'hsv_v': float(wandb.config.get('hsv_v', '0.4')) if cfg.dataset.augmentation else 0.0,
        'degrees': float(wandb.config.get('degrees', '0.0')) if cfg.dataset.augmentation else 0.0,
        'translate': float(wandb.config.get('translate', '0.1')) if cfg.dataset.augmentation else 0.0,
        'scale': float(wandb.config.get('scale', '0.5')) if cfg.dataset.augmentation else 0.0,
        'shear': float(wandb.config.get('shear', '0.0')) if cfg.dataset.augmentation else 0.0,
        'perspective': float(wandb.config.get('perspective', '0.0')) if cfg.dataset.augmentation else 0.0,
        'flipud': float(wandb.config.get('flipud', '0.0')) if cfg.dataset.augmentation else 0.0,
        'fliplr': float(wandb.config.get('fliplr', '0.5')) if cfg.dataset.augmentation else 0.0,
        'mosaic': float(wandb.config.get('mosaic', '1.0')) if cfg.dataset.augmentation else 0.0,
        'mixup': float(wandb.config.get('mixup', '0.0')) if cfg.dataset.augmentation else 0.0,

        # Loss weights (for YOLO's multi-component loss)
        'box': float(wandb.config.get('box', '7.5')),  # Bounding box loss
        'cls': float(wandb.config.get('cls', '0.5')),  # Classification loss
        'dfl': float(wandb.config.get('dfl', '1.5')),  # Distribution focal loss

        # Device
        'device': device,

        # Reproducibility
        'seed': cfg.dataset.random_seed,
        'deterministic': True,

        # Validation
        'val': True,
        'rect': not cfg.dataset.get('val_aug', 0),  # Rectangular val if val_aug=0

        # Saving
        'save': True,
        'save_period': 1 if cfg.opt.save_each_epoch else -1,
        'exist_ok': True,

        # Logging
        'project': model_dir,
        'name': 'training',
        'verbose': True,
        'plots': True,

        # Pretrained weights
        'pretrained': True,

        # Segmentation specific
        'overlap_mask': True,  # Allow mask overlap during training
        'mask_ratio': 4,  # Mask downsample ratio

        # Close mosaic augmentation in last N epochs
        'close_mosaic': 10,
    }

    # ========== START TRAINING ==========
    print(f"\n{'='*60}")
    print(f"Starting YOLOv8-seg training: {run.name}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Fold: {fold}")
    print(f"  - Model: {cfg.model.encoder_name}")
    print(f"  - Epochs: {train_args['epochs']}")
    print(f"  - Batch size: {train_args['batch']}")
    print(f"  - Learning rate: {train_args['lr0']}")
    print(f"  - Optimizer: {train_args['optimizer']}")
    print(f"  - Image size: {train_args['imgsz']}")
    print(f"  - Device: {train_args['device']}")
    print(f"  - Workers: {train_args['workers']}")
    print(f"  - Early stopping patience: {train_args['patience']}")
    print(f"  - Class weights: {cfg.opt.weights}")
    print(f"{'='*60}\n")

    try:
        # ========== TRAIN MODEL ==========
        results = model.train(**train_args)

        # ========== FINAL VALIDATION ==========
        print("\nRunning final validation...")
        metrics = model.val()

        # ========== LOG FINAL METRICS ==========
        # YOLOv8 segmentation metrics
        final_metrics = {
            'final/box_mAP50': metrics.box.map50,
            'final/box_mAP50-95': metrics.box.map,
            'final/seg_mAP50': metrics.seg.map50,
            'final/seg_mAP50-95': metrics.seg.map,
            'final/fitness': metrics.fitness,
        }

        # Log per-class metrics if available
        if hasattr(metrics.box, 'maps'):
            for i, map_val in enumerate(metrics.box.maps):
                final_metrics[f'final/class_{i}_box_mAP50-95'] = map_val

        if hasattr(metrics.seg, 'maps'):
            for i, map_val in enumerate(metrics.seg.maps):
                final_metrics[f'final/class_{i}_seg_mAP50-95'] = map_val

        wandb.log(final_metrics)

        # ========== LOG OPTIMIZATION METRIC ==========
        # Map the metric_name from cfg to corresponding YOLO metric
        metric_name = cfg.opt.metric_name
        if 'IoU' in metric_name or 'iou' in metric_name.lower():
            # Use seg mAP as proxy for IoU
            optimization_metric = metrics.seg.map
        elif 'mAP' in metric_name:
            optimization_metric = metrics.seg.map
        else:
            optimization_metric = metrics.fitness

        wandb.log({metric_name: optimization_metric})

        # ========== TRAINING SUMMARY ==========
        print(f"\n{'='*60}")
        print(f"Training completed successfully!")
        print(f"{'='*60}")
        print(f"Final Metrics:")
        print(f"  - Box mAP50-95: {metrics.box.map:.4f}")
        print(f"  - Box mAP50:    {metrics.box.map50:.4f}")
        print(f"  - Seg mAP50-95: {metrics.seg.map:.4f}")
        print(f"  - Seg mAP50:    {metrics.seg.map50:.4f}")
        print(f"  - Fitness:      {metrics.fitness:.4f}")
        print(f"  - {metric_name}: {optimization_metric:.4f}")
        print(f"{'='*60}\n")

        # ========== SAVE BEST MODEL ==========
        best_model_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            # Copy model to project directory
            import shutil
            dest_path = os.path.join(model_dir, 'best.pt')
            shutil.copy(str(best_model_path), dest_path)
            print(f"Best model saved to: {dest_path}")

            # Log model as wandb artifact
            artifact = wandb.Artifact(
                name=f"model-{run.id}",
                type="model",
                description=f"Best YOLOv8-seg model for fold {fold}",
                metadata={
                    'fold': fold,
                    'model_size': model_size,
                    'seg_mAP': float(metrics.seg.map),
                    'fitness': float(metrics.fitness),
                }
            )
            artifact.add_file(dest_path)
            run.log_artifact(artifact)
            print(f"Model artifact logged to wandb")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR during training!")
        print(f"{'='*60}")
        print(f"Error message: {str(e)}")
        print(f"{'='*60}\n")
        wandb.log({'error': str(e), 'training_failed': True})
        raise e

    finally:
        # Cleanup and finish wandb run
        print("Finishing wandb run...")
        wandb.finish()


if __name__ == "__main__":
    """
    Main entry point when running script directly.
    Normally this is called by wandb agent during sweeps.
    """
    train()