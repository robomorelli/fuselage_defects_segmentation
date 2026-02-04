#!/bin/bash

# Wrapper script to launch segmentation HPO sweeps on PBS cluster
# Usage: ./000_wrapper_run_batch_segmentation.sh --architecture deeplabv3plus --sweep_cfg screening --exps_num 2

# Default values
entity="robmorelli"
project_name="fuselage_segmentation_hpo"
architecture="deeplabv3plus"  # NEW: architecture parameter
sweep_cfg="screening"
exps_num=20
ngpus=1
ncpus=12
group="None"
sweep_id="None"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --entity)
      entity="$2"
      shift 2
      ;;
    --project_name)
      project_name="$2"
      shift 2
      ;;
    --architecture)
      architecture="$2"
      shift 2
      ;;
    --sweep_cfg)
      sweep_cfg="$2"
      shift 2
      ;;
    --exps_num)
      exps_num="$2"
      shift 2
      ;;
    --group)
      group="$2"
      shift 2
      ;;
    --ngpus)
      ngpus="$2"
      shift 2
      ;;
    --ncpus)
      ncpus="$2"
      shift 2
      ;;
    --sweep_id)
      sweep_id="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print the variables
echo "=========================================="
echo "Launching Segmentation HPO Sweep"
echo "=========================================="
echo "entity: $entity"
echo "project_name: $project_name"
echo "architecture: $architecture"
echo "sweep_cfg: $sweep_cfg"
echo "exps_num: $exps_num"
echo "group: $group"
echo "num gpus: $ngpus"
echo "num cpus: $ncpus"
echo "sweep_id: $sweep_id"
echo "=========================================="

source $HOME/.bashrc
cd /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/batch_scripts

python /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/batch_scripts/001_run_batch_hyp_segmentation.py \
  --entity "$entity" \
  --project_name "$project_name" \
  --architecture "$architecture" \
  --sweep_cfg "$sweep_cfg" \
  --exps_num "$exps_num" \
  --group "$group" \
  --ngpus "$ngpus" \
  --ncpus "$ncpus" \
  --sweep_id "$sweep_id"

# Usage examples:
#
# Screening (test all architectures):
# ./000_wrapper_run_batch_hyp_segmentation.sh --architecture deeplabv3plus --sweep_cfg screening --exps_num 2
#
# Resolution comparison:
# ./000_wrapper_run_batch_hyp_segmentation.sh --architecture deeplabv3plus --sweep_cfg resolution_comparison --exps_num 8
#
# Deep HPO on DeepLabV3+:
# ./000_wrapper_run_batch_hyp_segmentation.sh --architecture deeplabv3plus --sweep_cfg deeplabv3plus_hpo --exps_num 20
#
# Deep HPO on PSPNet:
# ./000_wrapper_run_batch_hyp_segmentation.sh --architecture pspnet --sweep_cfg pspnet_hpo --exps_num 20