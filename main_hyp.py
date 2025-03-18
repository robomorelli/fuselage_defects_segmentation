import argparse
import torch.optim
import wandb
from utils.general import read_yaml
from train import train
from box import Box
import subprocess
from config import *

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Paths to YAML files
    cfg_yaml = os.path.join(config_folder, f'{args.conf_yaml}.yaml')
    sweep_yaml = os.path.join(config_folder, f'{args.sweep_cfg}.yaml')

    # Read sweep configuration
    sweep_config = read_yaml(sweep_yaml)
    model_cfg = Box(read_yaml(cfg_yaml))

    metric_block = {
        'name': model_cfg.opt.metric_name,  # Metric you want to optimize
        'goal': model_cfg.opt.metric_goal  # Or 'maximize' depending on your goal
    }
    method = model_cfg.opt.method

    # Add the metric block to the sweep configuration
    sweep_config['metric'] = metric_block
    sweep_config['method'] = method
    sweep_config['parameters']['cfg'] = {'value': model_cfg}  # Add dataset YAML dynamically
    sweep_config['parameters']['project_name'] = {'value': args.project_name}  # Add model dynamically
    sweep_config['parameters']['ngpus'] = {'value': args.ngpus}
    sweep_config['parameters']['ncpus'] = {'value': args.ncpus}
    if args.group is not None:
        os.environ[ "WANDB_GROUP"] = args.group  # it cannot be added to sweep cfg because it need to init wandb before getting weep cfg (see train funct.)

    # Initialize the sweep
    # project is the name on wandb API
    if args.sweep_id is not None and args.sweep_id != "None":
        # Define the bash command
        #command = f"wandb agent robmorelli/segformer_hyp_opt_dv/exwq6fq6"
        command = f"wandb agent -p {args.project_name} -e {args.entity} {args.sweep_id}"
        # Run the command using subprocess
        subprocess.run(command, shell=True, check=True)
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project_name, entity=args.entity)
        print(f"SWEEP_ID: {sweep_id}")
        os.environ["sweep_id"] = sweep_id
        # Run the sweep
        wandb.agent(sweep_id, function=train, count=args.exps_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO sweep with wandb")
    parser.add_argument("--entity", default='robmorelli', help="Dataset configuration file (YAML)")
    parser.add_argument("--project_name", default='segformer_hyp_opt_zbook', help="Dataset configuration file (YAML)")
    parser.add_argument("--conf_yaml", default='segformer', help="Dataset configuration file (YAML)")
    parser.add_argument("--sweep_cfg", default='sweep', help="Sweep configuration file (YAML)")
    parser.add_argument("--ngpus", default=1, help="")
    parser.add_argument("--ncpus", default=6, help="")
    parser.add_argument("--exps_num", default=10, help="")
    parser.add_argument("--group", default=None, help="")
    parser.add_argument("--sweep_id", default=None, help="aa3wmwih")

    args, unknown = parser.parse_known_args()
    main(args)

