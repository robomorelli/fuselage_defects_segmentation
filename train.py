import torch
import torch.optim
from utils.training import training_cycle_segformer_multiclass
from utils.opt import EarlyStopping
import datetime
import wandb
from utils.training import load_model, create_dataloader
from config import *

def train():

    # Initialize wandb for this run
    group = os.getenv("WANDB_GROUP", None)
    if group is not None:
        run = wandb.init(group)
    else:
        run = wandb.init()

    cfg =  wandb.config.get('conf_yaml')
    cfg.dataset.fold = wandb.config.get('batch', 12)
    cfg.opt.weights =  wandb.config.get('classes_weights', 12)
    cfg.opt.lr = wandb.config.get('lr')
    cfg.opt.es_patience = wandb.config.get('es_patience', '7')
    cfg.opt.lr_patience = wandb.config.get('lr_patience', '3')

    # Get an environment variable
    device_env = os.getenv("GPU_ID")

    # Generate a unique timestamp for this training run
    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # Access the sweep ID for this run
    sweep_id = run.sweep_id
    api = wandb.Api()
    entity = run.entity  # Automatically get the entity of the current run
    project = run.project  # Automatically get the project of the current run
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    sweep_name = sweep.name  # Retrieve the sweep name

    # Define additional arguments
    batch_size = wandb.config.get('batch', 12)  # Default batch size if not in config
    device = wandb.config.get('devices', '0')

    if device_env:
        device = device_env

    # Configure CPU threading
    ncpus = int(wandb.config.get('ncpus', 0))
    if ncpus == 0:
        ncpus = int(device[-2])*12
    torch.set_num_threads(ncpus)
    num_workers = ncpus

    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["MKL_NUM_THREADS"] = str(ncpus)

    fold = wandb.config.get('fold', '1')  # Default batch size if not in config

    model_dir = os.path.join(wandb.config.get('project_name'), sweep_name, cfg.model.name, cfg.model.encoder_name, f"fold_{fold}"
                             , cfg.model.exp_name + f'_w_{cfg.opt.weight[0]}_{cfg.opt.weight[1]}_{cfg.opt.weight[2]}'+ "_" + now, run.name)

    #### INSTANTIATE MODEL ###
    model = load_model(cfg, wandb.config.get('unfreeze'))
    model.to(device)
    #### GET DATALOADERS ###
    train_dataloader, val_dataloader = create_dataloader(cfg, batch_size=batch_size, num_workers=num_workers)
    ### TRAIN ###

    if cfg.opt.crossentropy_loss:
        if cfg.opt.weights is not None:
            if cfg.opt.weights == 0:
                raise NotImplementedError
            weights = cfg.opt.weights
            weights = weights if isinstance(weights, torch.FloatTensor) else torch.FloatTensor(weights)
            if cfg.opt.ignore_index is not None:
                criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_idex=cfg.opt.ignore_index).to(
                    device)  # weight (Tensor, optional): a manual rescaling weight given
                # to each class as to be a Tensor of size `C` and floating point dtype
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=weights).to(
                    device)  # weight (Tensor, optional): a manual rescaling weight given
                # to each class as to be a Tensor of size `C` and floating point dtype
                print('crossentropy with pos weights')
        else:
            if cfg.opt.ignore_index is not None:
                criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.opt.ignore_index)
            else:
                criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss()
        print('BCE')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=cfg.opt.es_patience)

    training_cycle_segformer_multiclass(cfg=cfg, model=model, train_loader=train_dataloader,
                                      val_loader=val_dataloader,
                                      criterion=criterion, optimizer=optimizer
                                      , scheduler=scheduler, early_stopping=early_stopping,
                                      model_name=cfg.model.name,
                                      out_dir=model_dir, device=device,
                                      num_epochs=wandb.config.get('epochs'))

if __name__ == "__main__":
    train()