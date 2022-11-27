import os
import wandb
import argparse
from os.path import join, splitext

from trainer import trainer
from config.config import cfg_parser
from utils.utils import seed_everything
from utils.wandb_utils import init_wandb


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Cond. Independence Regularization for Causal Rep. Learning", 
        allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="default.yml",
        help="name of the config file to use"
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed"
        )
    parser.add_argument(
        "-w", "--wandb", action="store_true", help="Log to wandb"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default='online', help="wandb mode"
    )
    (args, unknown_args) = parser.parse_known_args()
    seed_everything(seed=args.seed, harsh=False)

    # Uncomment to provide GPU ID as input argument:
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version), args.seed)
    cfg["exp_cfg"].version = f'{splitext(args.version)[0]}/{args.seed}'
    cfg["exp_cfg"].run_name = cfg["exp_cfg"].version
    cfg["exp_cfg"].wandb = args.wandb

    if args.wandb:
        init_wandb(cfg.copy(), args.mode)
    
    pipeline = trainer.factory.create(cfg['model_cfg'].trainer_key, **cfg)
    pipeline.run()
    if args.wandb:
        wandb.finish()
