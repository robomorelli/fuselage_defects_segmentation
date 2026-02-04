import argparse
import torch.optim
import wandb
from utils.general import read_yaml
from trainer.train_mask2former import train
from box import Box
import subprocess
from config import *

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Paths to YAML files in configuration folder
    cfg_yaml = os.path.join('configuration', f'{args.conf_yaml}.yaml')
    sweep_yaml = os.path.join('configuration', f'{args.sweep_cfg}.yaml')

    # Read sweep configuration
    sweep_config = read_yaml(sweep_yaml)
    model_cfg = Box(read_yaml(cfg_yaml))

    metric_block = {
        'name': model_cfg.opt.metric_name,
        'goal': model_cfg.opt.metric_goal
    }
    method = model_cfg.opt.method

    # Add the metric block to the sweep configuration
    sweep_config['metric'] = metric_block
    sweep_config['method'] = method
    sweep_config['parameters']['cfg'] = {'value': model_cfg}
    sweep_config['parameters']['project_name'] = {'value': args.project_name}
    sweep_config['parameters']['ngpus'] = {'value': args.ngpus}
    sweep_config['parameters']['ncpus'] = {'value': args.ncpus}
    if args.group is not None:
        os.environ["WANDB_GROUP"] = args.group

    # Initialize the sweep
    if args.sweep_id is not None and args.sweep_id != "None":
        command = f"wandb agent -p {args.project_name} -e {args.entity} {args.sweep_id}"
        subprocess.run(command, shell=True, check=True)
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project_name, entity=args.entity)
        print(f"SWEEP_ID: {sweep_id}")
        os.environ["sweep_id"] = sweep_id
        wandb.agent(sweep_id, function=train, count=args.exps_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO sweep with wandb")
    parser.add_argument("--entity", default='robmorelli', help="Wandb entity")
    parser.add_argument("--project_name", default='segformer_hyp_opt_dv_3_classes_test', help="Wandb project name")
    parser.add_argument("--conf_yaml", default='mask2former', help="Model configuration file (YAML)")
    parser.add_argument("--sweep_cfg", default='sweep_mask2former', help="Sweep configuration file (YAML)")
    parser.add_argument("--ngpus", default=1, help="Number of GPUs")
    parser.add_argument("--ncpus", default=6, help="Number of CPUs")
    parser.add_argument("--exps_num", default=10, help="Number of experiments")
    parser.add_argument("--group", default=None, help="Wandb group name")
    parser.add_argument("--sweep_id", default=None, help="Existing sweep ID to resume")

    args, unknown = parser.parse_known_args()
    main(args)