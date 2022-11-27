# CIRCE
Implementation of Conditional Independence Regression CovariancE (CIRCE) for efficient conditional invariant representation learning.

## Environment
- Python 3.9.6
- Pytorch 1.12.0
- Torchvision 0.13.0
- Wandb 0.13.2

## Prepare data for training
```sh prepare_data.sh dsprites```

```sh prepare_data.sh yale-b```

## Training
To train a model, first create a .yml file in the config directory specifying the various data, model and training settings/hyperparameters. Refer to `config/default.yml` as an example.

Then execute ```python main.py -v <relative-path-to-yml-from-config> --seed <seed> -w```

Example: ```python main.py -v default.yml --seed 42 -w```

If you do not wish to sync to wandb while training add the option ```-m offline``` and sync anytime later with the `wandb sync` command.

If seed is not specified it will default to `0`.

Trained models are saved in the location specified in `experiment.output_location` in a subfolder named as per the seed. In wandb, experiments are logged under `<config name>/<seed>` in the `causal-rep` workspace.