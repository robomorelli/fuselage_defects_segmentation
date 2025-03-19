import argparse
import os

def main(args):
    # Define variables for entity, data_cfg, hyp_cfg, and sweep_cfg (replace these with actual values or arguments)
    entity = args.entity  # Replace with actual value or argument
    sweep_cfg = args.sweep_cfg  # Replace with actual value or argument
    project_name = args.project_name  # Replace with actual value or argument
    exps_num = args.exps_num  # Replace with actual value or argument
    ngpus = args.ngpus  # Replace with actual value or argument
    ncpus = args.ncpus  # Replace with actual value or argument

    group = None if args.group == "None" else args.group
    sweep_id = None if args.sweep_id == "None" else args.sweep_id

    # Generate PBS script
    pbs_script = \
f"""#!/bin/bash
#PBS -N fus_segmentation_hyp
#PBS -o fus_segmentation_hyp.txt
#PBS -q gpu
#PBS -e fus_segmentation_hyp.txt
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=1:ncpus={ncpus}:ngpus={ngpus},walltime=72:00:00

NUM_NODES=$(cat $PBS_NODEFILE | wc -l)

echo "start training"
# ENV that are the ARGS to pass to the function
echo "entity": {entity}
echo "sweep_cfg": {sweep_cfg}
echo "num ngpus": {ngpus}
echo "num ncpus": {ncpus}
echo "project_name": {project_name}
echo "group": {group}
echo "exps_num": {exps_num}
echo "sweep_id": {sweep_id}

module load proxy/proxy_20

source $HOME/.bashrc
cd /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation

conda activate flai
python /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/main_hyp.py --entity {entity} \
--project_name {project_name} --sweep_cfg {sweep_cfg} --ngpus {ngpus} --ncpus {ncpus} --exps_num {exps_num} \
--group {group} --sweep_id {sweep_id}
"""

    # Write the script to a file
    with open("./002_run_batch_hyp.pbs", "w") as f:
        f.write(pbs_script)

    # Submit the job
    os.system("qsub 002_run_batch_hyp.pbs")
    print(f"Job submitted with {args.ngpus} GPU(s).")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default='robmorelli', help="Dataset configuration file (YAML)")
    parser.add_argument("--project_name", default='hyperparameters_opt', help="Dataset configuration file (YAML)")
    parser.add_argument("--hyp_cfg", default='hyp', help="hyp config file name (YAML)")
    parser.add_argument("--sweep_cfg", default='sweep', help="Sweep configuration file (YAML)")
    parser.add_argument("--exps_num", default=10, help="number of trials")
    parser.add_argument("--group", default=None, help="number of trials")
    parser.add_argument('--ngpus', type=int, default=1, help="Number of GPUs to request for the job.")
    parser.add_argument('--ncpus', type=int, default=12, help="Number of GPUs to request for the job.")
    parser.add_argument('--sweep_id', default=None, help="")
    args, unknown = parser.parse_known_args()

    # Call the main function
    main(args)


