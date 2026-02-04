import argparse
import os

def main(args):
    """
    Generate and submit PBS script for segmentation HPO sweeps.
    """
    entity = args.entity
    sweep_cfg = args.sweep_cfg
    project_name = args.project_name
    architecture = args.architecture  # NEW
    exps_num = args.exps_num
    ngpus = args.ngpus
    ncpus = args.ncpus

    group = None if args.group == "None" else args.group
    sweep_id = None if args.sweep_id == "None" else args.sweep_id

    # Generate PBS script
    pbs_script = \
f"""#!/bin/bash
#PBS -N seg_{architecture}_hpo
#PBS -o seg_{architecture}_hpo.txt
#PBS -q gpu
#PBS -e seg_{architecture}_hpo.txt
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=1:ncpus={ncpus}:ngpus={ngpus},walltime=72:00:00

NUM_NODES=$(cat $PBS_NODEFILE | wc -l)

echo "=========================================="
echo "Starting Segmentation HPO Training"
echo "=========================================="
echo "entity: {entity}"
echo "architecture: {architecture}"
echo "sweep_cfg: {sweep_cfg}"
echo "num ngpus: {ngpus}"
echo "num ncpus: {ncpus}"
echo "project_name: {project_name}"
echo "group: {group}"
echo "exps_num: {exps_num}"
echo "sweep_id: {sweep_id}"
echo "=========================================="

module load proxy/proxy_20

source $HOME/.bashrc
cd /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation

conda activate flai
python /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/main_hyp_segmentation.py \\
--entity {entity} \\
--project_name {project_name} \\
--architecture {architecture} \\
--sweep_cfg {sweep_cfg} \\
--ngpus {ngpus} \\
--ncpus {ncpus} \\
--exps_num {exps_num} \\
--group {group} \\
--sweep_id {sweep_id}
"""

    # Write the script to a file
    pbs_filename = f"002_run_batch_{architecture}_hpo.pbs"
    with open(pbs_filename, "w") as f:
        f.write(pbs_script)

    # Submit the job
    os.system(f"qsub {pbs_filename}")
    print(f"\n{'='*50}")
    print(f"✅ PBS Job submitted!")
    print(f"   Architecture: {architecture}")
    print(f"   Sweep config: {sweep_cfg}")
    print(f"   Experiments: {exps_num}")
    print(f"   GPUs: {ngpus}, CPUs: {ncpus}")
    print(f"   PBS file: {pbs_filename}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Submit segmentation HPO sweeps to PBS cluster')
    parser.add_argument("--entity", default='robmorelli', help="Wandb entity")
    parser.add_argument("--project_name", default='fuselage_segmentation_hpo', help="Wandb project name")
    parser.add_argument("--architecture", default='deeplabv3plus',
                       choices=['deeplabv3plus', 'pspnet', 'segformer', 'mask2former'],
                       help="Model architecture to train")
    parser.add_argument("--sweep_cfg", default='screening', help="Sweep configuration file name (without .yaml)")
    parser.add_argument("--exps_num", type=int, default=10, help="Number of experiments to run")
    parser.add_argument("--group", default=None, help="Wandb group name")
    parser.add_argument('--ngpus', type=int, default=1, help="Number of GPUs")
    parser.add_argument('--ncpus', type=int, default=12, help="Number of CPUs")
    parser.add_argument('--sweep_id', default=None, help="Existing sweeps ID (optional)")
    args, unknown = parser.parse_known_args()

    # Call the main function
    main(args)