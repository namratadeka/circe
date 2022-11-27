import wandb


def init_wandb(cfg: dict, mode='online') -> None:
    """Initialize project on Weights & Biases
    Args:
        cfg (dict): Configuration dictionary
    """
    for key in cfg:
        cfg[key] = cfg[key].__dict__

    run = wandb.init(project="circe", name=cfg["exp_cfg"]["run_name"], 
               notes=cfg["exp_cfg"]["description"], config=cfg, mode=mode)
    print(f'Wandb run: {run.id}')

def log_epoch_summary(epochID:int, mode:str, losses:dict):
    logs = {}
    for key in losses.keys():
        logs.update({"{}/{}".format(mode, key): losses[key]})

    wandb.log(logs, step=epochID)
