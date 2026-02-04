#!/bin/bash

# Default values
entity="robmorelli"
project_name="segformer_hyp_opt_dv_3_classes"
sweep_cfg="sweep"
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
echo "entity: $entity"
echo "project_name: $project_name"
echo "sweep_cfg: $sweep_cfg"
echo "exps_num: $exps_num"
echo "group: $group"
echo "num gpus: $ngpus"
echo "num cpus: $ncpus"
echo "num sweep_id: $sweep_id"

source $HOME/.bashrc
cd /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/batch_scripts

python /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/batch_scripts/001_run_batch_hyp.py \
  --entity "$entity" \
  --project_name "$project_name" \
  --sweep_cfg "$sweep_cfg" \
  --exps_num "$exps_num" \
  --group "$group" \
  --ngpus "$ngpus" \
  --ncpus "$ncpus" \
  --sweep_id "$sweep_id"


#./artificial_intelligence/repos/fuselage_defects_segmentation/batch_scripts/000_wrapper_run_batch_hyp.sh --entity robmorelli --project_name segformer_hyp_opt_dv_3_classes  --sweep_cfg sweeps --exps_num 20  --ngpus 3 --ncpus 36

