"""
Main entry point for segmentation HPO sweeps.
Supports multiple architectures: DeepLabV3+, PSPNet, SegFormer, Mask2Former
"""

import argparse
import wandb
import yaml
import os
from box import Box

# Import the training function
from trainer.train_segmentator import train


def main():
    """
    Main function to launch hyperparameter optimization sweeps.
    """
    parser = argparse.ArgumentParser(description='Segmentation HPO Sweep')
    parser.add_argument('--entity', type=str, default='robmorelli', help='Wandb entity')
    parser.add_argument('--project_name', type=str, default='fuselage_segmentation_hpo_pspnet', help='Wandb project name')
    parser.add_argument('--sweep_cfg', type=str, default='sweep_pspnet', help='Sweep config file (without .yaml) choices=[sweep_deeplabv3plus, sweep_pspnet]')
    #parser.add_argument('--architecture', type=str, default='pspnet',
    #                    choices=['deeplabv3plus', 'pspnet'], #'mask2former'],
    #                    help='Model architecture')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--ncpus', type=int, default=12, help='Number of CPUs')
    parser.add_argument('--exps_num', type=int, default=1, help='Number of experiments')
    parser.add_argument('--group', type=str, default=None, help='Wandb group name')
    parser.add_argument('--sweep_id', type=str, default=None, help='Existing sweeps ID')

    args = parser.parse_args()

    # Load sweeps configuration
    sweep_config_path = f'configuration/sweeps/{args.sweep_cfg}.yaml'
    if not os.path.exists(sweep_config_path):
        raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")

    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Update sweeps config with runtime parameters
    sweep_config['parameters']['n_gpus'] = {'value': args.ngpus}
    sweep_config['parameters']['ncpus'] = {'value': args.ncpus}

    # If architecture is specified and not in sweeps config, add it
    if 'architecture' not in sweep_config['parameters']:
        sweep_config['parameters']['architecture'] = {'value': args.architecture}

    architecture = sweep_config['parameters']['architecture']


    print("\n" + "=" * 60)
    print("SEGMENTATION HPO SWEEP")
    print("=" * 60)
    print(f"Entity: {args.entity}")
    print(f"Project: {args.project_name}")
    print(f"Architecture: {architecture}")
    print(f"Sweep config: {args.sweep_cfg}")
    print(f"Experiments: {args.exps_num}")
    print(f"Resources: {args.ngpus} GPU(s), {args.ncpus} CPU(s)")
    print("=" * 60 + "\n")


    # Create or use existing sweeps
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"✓ Using existing sweeps: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project_name,
            entity=args.entity
        )
        print(f"✓ Created new sweeps: {sweep_id}")

    print(f"\n🚀 Starting {args.exps_num} experiment(s)...")
    print(f"   Sweep URL: https://wandb.ai/{args.entity}/{args.project_name}/sweeps/{sweep_id}\n")

    # Run the sweeps agent
    wandb.agent(
        sweep_id,
        function=train,
        count=args.exps_num,
        project=args.project_name,
        entity=args.entity
    )

    print("\n" + "=" * 60)
    print("✅ SWEEP COMPLETED!")
    print("=" * 60)
    print(f"View results: https://wandb.ai/{args.entity}/{args.project_name}/sweeps/{sweep_id}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()