import os
import torch
import torch.optim
from utils.training_mask2former import (
    training_cycle_mask2former,
    load_mask2former_model,
    create_mask2former_dataloader
)
from utils.opt import EarlyStopping
from datetime import datetime
import wandb
from box import Box
from config import *
import subprocess


def get_gpu_memory():
    """Returns the memory usage of all available GPUs."""
    gpu_info = []
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,nounits,noheader'])
        result = result.decode('utf-8').strip().split('\n')
        for line in result:
            parts = line.split(', ')
            gpu_index = int(parts[0])
            memory_free = int(parts[1])
            memory_total = int(parts[2])
            memory_used = memory_total - memory_free
            gpu_info.append((gpu_index, memory_used, memory_free, memory_total))
    except Exception as e:
        print(f"Error retrieving GPU memory information: {e}")
    return gpu_info


def select_gpus(n=2):
    """Selects the N GPUs with the most free memory."""
    gpu_info = get_gpu_memory()
    if not gpu_info:
        raise ValueError("No GPU information found.")

    sorted_gpus = sorted(gpu_info, key=lambda x: x[2], reverse=True)
    selected_gpus = [gpu[0] for gpu in sorted_gpus[:n]]
    print(f"Selected GPUs based on memory: {selected_gpus}")
    return selected_gpus


def train():
    # Initialize wandb
    group = os.getenv("WANDB_GROUP", None)
    if group is not None:
        run = wandb.init(group=group)
    else:
        run = wandb.init()

    # Load config
    cfg = Box(wandb.config.get('cfg'))
    if not cfg.wandb.online:
        os.environ["WANDB_MODE"] = "offline"

    # Parse wandb config
    cfg.dataset.fold = int(wandb.config.get('fold', '1'))
    cfg.opt.weights = list(wandb.config.get('classes_weights', [1, 1, 1]))
    cfg.dataset.crop_size = int(wandb.config.get('crop_size', '512'))  # ADD THIS
    cfg.opt.lr = float(wandb.config.get('lr', '0.0001'))
    cfg.opt.weight_decay = float(wandb.config.get('weight_decay', '0.0001'))
    cfg.opt.es_patience = int(wandb.config.get('es_patience', '7'))
    cfg.opt.lr_patience = int(wandb.config.get('lr_patience', '3'))
    cfg.model.backbone = str(wandb.config.get('backbone', 'swin_tiny'))
    cfg.model.num_queries = int(wandb.config.get('num_queries', '30'))
    cfg.opt.mask_weight = float(wandb.config.get('mask_weight', '5.0'))
    cfg.opt.dice_weight = float(wandb.config.get('dice_weight', '5.0'))

    summary_dict = {'maximize': "max", "minimize": 'min'}
    wandb.define_metric(cfg.opt.metric_name, summary=summary_dict[cfg.opt.metric_goal] + ',last')

    # Generate timestamp and directory structure
    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    sweep_id = run.sweep_id
    api = wandb.Api()
    entity = run.entity
    project = run.project
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    sweep_name = sweep.name
    fold = int(wandb.config.get('fold', '1'))

    weights_str = '_'.join([str(w) for w in cfg.opt.weights])
    model_dir = os.path.join(
        wandb.config.get('project_name'),
        sweep_name,
        cfg.model.name,
        cfg.model.backbone,
        f"fold_{fold}",
        f"{cfg.model.exp_name}_q{cfg.model.num_queries}_w{weights_str}_{now}",
        run.name
    )

    cfg.model.checkpoint = os.path.join(model_dir, cfg.model.name + '_best.pth')
    os.makedirs(model_dir, exist_ok=True)

    # GPU setup
    device_env = os.getenv("GPU_ID")
    n_gpus = int(wandb.config.get('n_gpus', '1'))

    if device_env:
        device_env = int(device_env)
        device = f"cuda:{device_env}"
        device_ids = [device_env]
    else:
        print(f'Using {n_gpus} GPU(s)')
        selected_gpus = select_gpus(n=n_gpus)
        device_ids = list(range(0, n_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        device = f"cuda:{device_ids[0]}"

    # CPU threading
    ncpus = int(wandb.config.get('ncpus', '0'))
    if ncpus == 0:
        ncpus = n_gpus * 12
    torch.set_num_threads(ncpus)
    num_workers = ncpus
    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["MKL_NUM_THREADS"] = str(ncpus)

    # Adjust batch size
    batch_size = int(wandb.config.get('batch', '2')) * n_gpus

    # Load model
    print("\n" + "=" * 50)
    print("Loading Mask2Former model...")
    print("=" * 50)
    model = load_mask2former_model(cfg, device)

    # Multi-GPU setup
    if n_gpus > 1:
        print(f"Using DataParallel with GPUs: {device_ids}")
        model = model.to(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model.to(device)

    # DataLoader
    print("\n" + "=" * 50)
    print("Creating dataloaders...")
    print("=" * 50)
    train_dataloader, val_dataloader = create_mask2former_dataloader(
        cfg, batch_size=batch_size, num_workers=num_workers
    )

    print(f"✓ Train batches: {len(train_dataloader)}")
    print(f"✓ Val batches: {len(val_dataloader)}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.opt.lr,
        weight_decay=cfg.opt.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        factor=0.8,
        patience=cfg.opt.lr_patience,
        threshold=0.0001,
        min_lr=9e-8,
        verbose=True
    )

    early_stopping = EarlyStopping(patience=cfg.opt.es_patience)

    # Training
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    training_cycle_mask2former(
        cfg=cfg,
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        model_name=cfg.model.name,
        out_dir=model_dir,
        device=device,
        num_epochs=int(wandb.config.get('epochs', '100')),
        metric_goal=cfg.opt.metric_goal
    )


if __name__ == "__main__":
    train()