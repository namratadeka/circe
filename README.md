# CIRCE
Implementation of Conditional Independence Regression CovariancE (CIRCE) from the paper [Efficient Conditionally Invariant Representation Learning](https://openreview.net/forum?id=dJruFeSRym1). This repository also contains code for [HSCIC](https://arxiv.org/abs/2207.09768) and [GCM](https://arxiv.org/abs/1804.07203).

## Environment
- Python 3.9.6
- Pytorch 1.12.0
- Torchvision 0.13.0
- Wandb 0.13.2

## Prepare data for training
```sh prepare_data.sh dsprites```

```sh prepare_data.sh yale-b```

## Training
To train a model, first create a .yml file in the config directory specifying the various data, model and training settings/hyperparameters. Refer to `config` for examples.

Then execute ```python main.py -v <relative-path-to-yml-from-config> --seed <seed> -w```

Example: ```python main.py -v dsprites_linear/circe.yml --seed 42 -w```

If you do not wish to sync to wandb while training add the option ```-m offline``` and sync anytime later with the `wandb sync` command.

If seed is not specified it will default to `0`.

Trained models are saved in the location specified in `experiment.output_location` in a subfolder named as per the seed. In wandb, experiments are logged under `<config file name>/<seed>` in the `circe` workspace.
