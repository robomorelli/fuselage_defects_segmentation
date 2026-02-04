import sys

sys.path.append('.')

import wandb
import torch
import torch.nn as nn
import os
from box import Box
import subprocess

from utils.training_segmentation import (
    load_segmentation_model,
    create_segmentation_dataloader,
    training_cycle_segmentation
)
from utils.opt import EarlyStopping


def get_free_gpu():
    """Get GPU with most free memory."""
    try:
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"],
            encoding='utf-8'
        )
        gpu_memory = [int(x.split()[0]) for i, x in enumerate(gpu_stats.split('\n')) if i > 0 and x]
        return gpu_memory.index(max(gpu_memory))
    except:
        return 0


def set_gpu(gpus=None, n_gpus=1):
    """
    Set GPU devices for training.

    Args:
        gpus: List of GPU IDs or None (auto-select)
        n_gpus: Number of GPUs to use

    Returns:
        device: torch.device
        gpu_ids: List of GPU IDs
    """
    if gpus is None:
        # Auto-select GPUs with most free memory
        gpu_ids = []
        for _ in range(n_gpus):
            free_gpu = get_free_gpu()
            gpu_ids.append(free_gpu)

        print(f"Auto-selected GPUs: {gpu_ids}")
    else:
        gpu_ids = gpus if isinstance(gpus, list) else [gpus]
        print(f"Using specified GPUs: {gpu_ids}")

    if len(gpu_ids) > 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        device = torch.device(f'cuda:0')
        print(f"Training on GPU(s): {gpu_ids}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        device = torch.device('cpu')
        print("Training on CPU")

    return device, gpu_ids


def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")


def train():
    """Main training function for segmentation HPO."""

    # Initialize wandb
    run = wandb.init()

    # Get architecture from wandb config
    architecture = wandb.config.get('architecture', 'deeplabv3plus')

    # Load architecture-specific config
    config_path = f'configuration/architectures/{architecture}.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = Box.from_yaml(filename=config_path)

    # Check wandb online/offline mode
    wandb_online = cfg.get('wandb', {}).get('online', 1)
    if not wandb_online:
        os.environ["WANDB_MODE"] = "offline"

    # Override with wandb sweeps parameters
    cfg.dataset.fold = int(wandb.config.get('fold', 1))
    cfg.dataset.crop_size = int(wandb.config.get('crop_size', 512))
    cfg.dataset.batch_size = int(wandb.config.get('batch', 4))

    cfg.model.architecture = str(wandb.config.get('architecture', 'deeplabv3plus.yaml'))
    cfg.model.backbone = str(wandb.config.get('backbone', 'resnet50'))
    cfg.model.freeze_layers = wandb.config.get("freeze_layers", 0)

    cfg.opt.weights = list(wandb.config.get('classes_weights', [1, 1, 1.05, 2.5]))
    cfg.opt.lr = float(wandb.config.get('lr', 0.0001))
    cfg.opt.weight_decay = float(wandb.config.get('weight_decay', 0.0001))
    cfg.opt.lr_patience = int(wandb.config.get('lr_patience', 5))
    cfg.opt.es_patience = int(wandb.config.get('es_patience', 7))
    cfg.opt.epochs = int(wandb.config.get('epochs', 100))

    # Mask2Former specific parameters (if applicable)
    if cfg.model.architecture == 'mask2former':
        cfg.model.num_queries = int(wandb.config.get('num_queries', 30))
        cfg.opt.mask_weight = float(wandb.config.get('mask_weight', 5.0))
        cfg.opt.dice_weight = float(wandb.config.get('dice_weight', 5.0))
        cfg.opt.cls_weight = float(wandb.config.get('cls_weight', 2.0))

    # Add tags for wandb organization
    wandb.run.tags = wandb.run.tags + (architecture, cfg.model.backbone)

    # GPU setup
    n_gpus = int(wandb.config.get('n_gpus', 1))
    device, gpu_ids = set_gpu(gpus=None, n_gpus=n_gpus)

    print(f"\nUsing device: {device}")
    get_gpu_memory()

    print("\n" + "=" * 50)
    print(f"Training {cfg.model.architecture.upper()} with {cfg.model.backbone}")
    print(f"Fold: {cfg.dataset.fold}")
    print(f"Crop size: {cfg.dataset.crop_size}")
    print(f"Batch size: {cfg.dataset.batch_size}")
    print(f"Learning rate: {cfg.opt.lr}")
    print(f"Class weights: {cfg.opt.weights}")
    print("=" * 50)

    # Load model
    print("\n" + "=" * 50)
    print("Loading model...")
    print("=" * 50)
    model = load_segmentation_model(cfg, device)
    model = model.to(device)

    # Multi-GPU support
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on {len(gpu_ids)} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))

    print(f"✓ Model loaded successfully")
    get_gpu_memory()

    # Create dataloaders
    print("\n" + "=" * 50)
    print("Creating dataloaders...")
    print("=" * 50)
    train_loader, val_loader = create_segmentation_dataloader(
        cfg,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.opt.num_workers
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.opt.lr,
        weight_decay=cfg.opt.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=cfg.opt.lr_patience,
        verbose=True
    )

    early_stopping = EarlyStopping(
        patience=cfg.opt.es_patience,
        min_delta=0.0001
    )

    # Training
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    model_name = f"{cfg.model.architecture}_{cfg.model.backbone}_fold{cfg.dataset.fold}"
    out_dir = f"./models/{cfg.model.architecture}"

    training_cycle_segmentation(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        model_name=model_name,
        out_dir=out_dir,
        device=device,
        num_epochs=cfg.opt.epochs,
        metric_goal=cfg.opt.metric_goal
    )


if __name__ == "__main__":
    train()