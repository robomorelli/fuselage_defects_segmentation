import torch
import torch.optim
from utils.training import training_cycle_segformer_multiclass
from utils.opt import EarlyStopping
from datetime import datetime
import wandb
from box import Box
from utils.training import load_model, create_dataloader
from config import *
import subprocess
import numpy as np

def get_gpu_memory():
    """
    Returns the memory usage of all available GPUs on the node.
    Uses `nvidia-smi` to get the GPU memory stats.
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
                (gpu_index, memory_used, memory_free, memory_total))  # (index, used memory, free memory, total memory)
    except Exception as e:
        print(f"Error retrieving GPU memory information: {e}")
    return gpu_info


def select_gpus(n=2):
    """
    Selects the N GPUs with the lowest memory usage (or highest available memory) for training.
    :param n: Number of GPUs to select.
    :return: A list of selected GPU ids.
    """
    # Get GPU memory information
    gpu_info = get_gpu_memory()
    if not gpu_info:
        raise ValueError("No GPU information found.")

    # Sort GPUs by free memory (descending), prioritize the ones with most free memory
    sorted_gpus = sorted(gpu_info, key=lambda x: x[2], reverse=True)  # Sort by free memory

    # Select the top N GPUs
    print(sorted_gpus, n)
    selected_gpus = [gpu[0] for gpu in sorted_gpus[:n]]  # Select the top N GPUs

    print(f"Selected GPUs based on memory: {selected_gpus}")
    return selected_gpus


def train():
    # Initialize wandb for this run
    group = os.getenv("WANDB_GROUP", None)
    if group is not None:
        run = wandb.init(group)
    else:
        run = wandb.init()

    # Explicitly cast config values to the appropriate types
    cfg = Box(wandb.config.get('cfg'))
    if not cfg.wandb.online:
        os.environ["WANDB_MODE"] = "offline"  # Run wandb offline
    cfg.dataset.fold = int(wandb.config.get('fold', '1'))  # Convert fold to int
    cfg.opt.weights = list(wandb.config.get('classes_weights'))
    cfg.opt.lr = float(wandb.config.get('lr', '0.0003'))  # Convert lr to float
    cfg.opt.es_patience = int(wandb.config.get('es_patience', '7'))  # Convert es_patience to int
    cfg.opt.lr_patience = int(wandb.config.get('lr_patience', '3'))  # Convert lr_patience to int
    cfg.model.encoder_name = str(wandb.config.get('encoder_names'))  # Convert lr_patience to int

    summary_dict = {'maximize':"max", "minimize":'min'}
    wandb.define_metric(cfg.opt.metric_name, summary=summary_dict[cfg.opt.metric_goal]+',last')

    # Generate a unique timestamp for this training run
    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # Access the sweep ID for this run
    sweep_id = run.sweep_id
    api = wandb.Api()
    entity = run.entity  # Automatically get the entity of the current run
    project = run.project  # Automatically get the project of the current run
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    sweep_name = sweep.name  # Retrieve the sweep name

    fold = int(wandb.config.get('fold', '1'))  # Ensure fold is an integer

    model_dir = os.path.join(wandb.config.get('project_name'), sweep_name, cfg.model.name, cfg.model.encoder_name,
                             f"fold_{fold}", cfg.model.exp_name + f'_w_{cfg.opt.weights[0]}_{cfg.opt.weights[1]}_{cfg.opt.weights[2]}' + "_" + now,
                             run.name)

    #### INSTANTIATE MODEL ###
    model = load_model(cfg, wandb.config.get('unfreeze'))

    # Get an environment variable for device selection (if provided)
    device_env = os.getenv("GPU_ID")
    if device_env:
        device_env = int(device_env)  # Convert GPU_ID to an integer
        device = f"cuda:{device_env}"
        print(f"device {device}")
        model = model.to(device)  # Move model to first selected GPU
    else:
        # Select the N GPUs with the least load (top N)
        n_gpus = int(wandb.config.get('ngpus'))  # Convert devices to integer
        print(f'gpu numbers {n_gpus}')
        selected_gpus = select_gpus(n=n_gpus)
        # Convert selected GPUs into a format suitable for PyTorch DataParallel
        device_ids = selected_gpus
        #device_ids = list(range(0, n_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

        # If using multiple GPUs, use DataParallel
        if len(device_ids) > 1:
            print(f"Using GPUs {device_ids} for training")
            model = model.to(device_ids[0])
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            print(f"device_ids {device_ids}")
            model = model.to(f"cuda:{device_ids[0]}")  # Move model to first selected GPU
        device = f"cuda:{device_ids[0]}"  # Set the device to the first GPU for criterion and other tasks

    # Configure CPU threading
    ncpus = int(wandb.config.get('ncpus', '0'))  # Ensure ncpus is an integer
    if ncpus == 0:
        ncpus = int(device[-2]) * 12
    torch.set_num_threads(ncpus)
    num_workers = ncpus

    # Adjust batch size based on the number of GPUs
    batch_size = int(wandb.config.get('batch', '12')) * n_gpus  # Convert batch to int and adjust based on GPUs

    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["MKL_NUM_THREADS"] = str(ncpus)

    # Get DataLoader
    train_dataloader, val_dataloader = create_dataloader(cfg, batch_size=batch_size, num_workers=num_workers)

    ### LOSS FUNCTION ###
    if cfg.opt.crossentropy_loss:
        if cfg.opt.weights is not None:
            weights = cfg.opt.weights
            weights = weights if isinstance(weights, torch.FloatTensor) else torch.FloatTensor(weights)
            if cfg.opt.ignore_index is not None:
                criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=cfg.opt.ignore_index).to(device)
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        else:
            if cfg.opt.ignore_index is not None:
                criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.opt.ignore_index).to(device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.BCELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8,
                                                           patience=cfg.opt.lr_patience,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=cfg.opt.es_patience)

    # Start training
    training_cycle_segformer_multiclass(cfg=cfg, model=model, train_loader=train_dataloader,
                                        val_loader=val_dataloader,
                                        criterion=criterion, optimizer=optimizer,
                                        scheduler=scheduler, early_stopping=early_stopping,
                                        model_name=cfg.model.name, out_dir=model_dir,
                                        device=device,
                                        num_epochs=int(wandb.config.get('epochs', '10')),
                                        metric_goal=cfg.opt.metric_goal)  # Ensure epochs is an int


if __name__ == "__main__":
    train()
