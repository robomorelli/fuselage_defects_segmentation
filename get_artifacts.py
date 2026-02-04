import wandb

import argparse

def main(args)
    api = wandb.Api()
    sweep = api.sweep(f"{}/{}/<sweep_id>")
    runs = sorted(sweep.runs,
      key=lambda run: run.summary.get("val_acc", 0), reverse=True)
    val_acc = runs[0].summary.get("val_acc", 0)
    print(f"Best run {runs[0].name} with {val_acc}% validation accuracy")

    runs[0].file("model.h5").download(replace=True)
    print("Best model saved to model-best.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO sweeps with wandb")
    parser.add_argument("--entity", default='robmorelli', help="Dataset configuration file (YAML)")
    parser.add_argument("--project_name", default='segformer_hyp_opt_dv_3_classes_test', help="Dataset configuration file (YAML)")
    parser.add_argument("--sweep_id", default='sweeps', help="Sweep configuration file (YAML)")
    parser.add_argument("--metric", default="Validation IoU", help="")
    parser.add_argument("--rank", default=None, help="aa3wmwih")

    args, unknown = parser.parse_known_args()
    main(args)
