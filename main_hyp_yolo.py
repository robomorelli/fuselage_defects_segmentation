"""
YOLOv8-seg hyperparameter optimization sweep launcher
Mirror of main_hyp.py for Segformer but adapted for YOLO
"""

import argparse
import torch.optim
import wandb
from utils.general import read_yaml
from trainer.train_yolo import train  # Import train function from train_yolo.py
from box import Box
import subprocess
from config import *

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    """
    Main function to initialize and run wandb sweep for YOLOv8-seg.

    Args:
        args: Command line arguments containing sweep configuration
    """
    # Paths to YAML files
    cfg_yaml = os.path.join(config_folder, f'{args.conf_yaml}.yaml')
    sweep_yaml = os.path.join(config_folder, f'{args.sweep_cfg}.yaml')

    # Read sweep configuration
    sweep_config = read_yaml(sweep_yaml)
    model_cfg = Box(read_yaml(cfg_yaml))

    # Create metric block for optimization
    metric_block = {
        'name': model_cfg.opt.metric_name,  # Metric you want to optimize
        'goal': model_cfg.opt.metric_goal  # 'maximize' or 'minimize'
    }
    method = model_cfg.opt.method

    # Add the metric block to the sweep configuration
    sweep_config['metric'] = metric_block
    sweep_config['method'] = method

    # Add dynamic parameters to sweep config
    sweep_config['parameters']['cfg'] = {'value': model_cfg}  # Add model config dynamically
    sweep_config['parameters']['project_name'] = {'value': args.project_name}  # Add project name
    sweep_config['parameters']['ngpus'] = {'value': args.ngpus}  # Number of GPUs
    sweep_config['parameters']['ncpus'] = {'value': args.ncpus}  # Number of CPUs

    # Set wandb group if provided
    # Note: Cannot be added to sweep cfg because wandb needs to init before getting sweep cfg
    if args.group is not None:
        os.environ["WANDB_GROUP"] = args.group

    # Initialize the sweep
    # project is the name on wandb API
    if args.sweep_id is not None and args.sweep_id != "None":
        # Resume existing sweep using sweep ID
        print(f"\n{'=' * 60}")
        print(f"RESUMING EXISTING SWEEP")
        print(f"{'=' * 60}")
        print(f"Sweep ID: {args.sweep_id}")
        print(f"Project:  {args.project_name}")
        print(f"Entity:   {args.entity}")
        print(f"{'=' * 60}\n")

        # Define the bash command to resume sweep
        command = f"wandb agent -p {args.project_name} -e {args.entity} {args.sweep_id}"

        # Run the command using subprocess
        subprocess.run(command, shell=True, check=True)
    else:
        # Create new sweep
        print(f"\n{'=' * 60}")
        print(f"CREATING NEW SWEEP")
        print(f"{'=' * 60}")
        print(f"Project:     {args.project_name}")
        print(f"Entity:      {args.entity}")
        print(f"Config:      {args.conf_yaml}")
        print(f"Sweep cfg:   {args.sweep_cfg}")
        print(f"Method:      {method}")
        print(f"Metric:      {model_cfg.opt.metric_name} ({model_cfg.opt.metric_goal})")
        print(f"Experiments: {args.exps_num}")
        if args.group:
            print(f"Group:       {args.group}")
        print(f"{'=' * 60}\n")

        sweep_id = wandb.sweep(sweep_config, project=args.project_name, entity=args.entity)

        print(f"\n{'=' * 60}")
        print(f"SWEEP CREATED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(f"SWEEP_ID: {sweep_id}")
        print(f"\nTo resume this sweep later, use:")
        print(f"python main_hyp_yolo.py --sweep_id {sweep_id} --project_name {args.project_name}")
        print(f"{'=' * 60}\n")

        os.environ["sweep_id"] = sweep_id

        # Run the sweep
        print(f"Starting sweep agent with {args.exps_num} experiments...\n")
        wandb.agent(sweep_id, function=train, count=args.exps_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLOv8-seg hyperparameter sweep with wandb")

    # Wandb configuration
    parser.add_argument("--entity", default='robmorelli',
                        help="Wandb entity name")
    parser.add_argument("--project_name", default='yolov8seg_hyp_opt_dv_3_classes',
                        help="Wandb project name")

    # Configuration files
    parser.add_argument("--conf_yaml", default='yolo',
                        help="Model configuration file (YAML, without extension)")
    parser.add_argument("--sweep_cfg", default='sweep_yolo',
                        help="Sweep configuration file (YAML, without extension)")

    # Hardware configuration
    parser.add_argument("--ngpus", type=int, default=1,
                        help="Number of GPUs to use per experiment")
    parser.add_argument("--ncpus", type=int, default=6,
                        help="Number of CPU threads to use per experiment")

    # Sweep configuration
    parser.add_argument("--exps_num", type=int, default=10,
                        help="Number of experiments to run in sweep")
    parser.add_argument("--group", default=None,
                        help="Wandb group name for organizing runs")
    parser.add_argument("--sweep_id", default=None,
                        help="Existing sweep ID to resume (e.g., 'abc123xyz')")

    args, unknown = parser.parse_known_args()
    main(args)