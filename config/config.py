import os
import yaml
from os.path import join, basename, splitext


class Config(object):
    """
    Class for all attributes and functionalities related to a particular training run.
    """
    def __init__(self, cfg_file: str, params: dict):
        self.cfg_file = cfg_file
        self.__dict__.update(params)

def cfg_parser(cfg_file: str, seed: int, sweep: bool = False) -> dict:
    """
    This functions reads an input config file and instantiates objects of
    Config types.
    args:
        cfg_file (string): path to cfg file
    returns:
        data_cfg (Config)
        model_cfg (Config)
        exp_cfg (Config)
    """
    cfg = yaml.load(open(cfg_file, "r"), Loader=yaml.FullLoader)

    exp_cfg = Config(cfg_file, cfg["experiment"])
    dir_dict = dir_util(exp_cfg, seed, sweep)
    exp_cfg.__dict__.update(dir_dict)

    data_cfg = Config(cfg_file, cfg["data"])
    model_cfg = Config(cfg_file, cfg["model"])
    # default HSIC is unbiased
    if not hasattr(model_cfg, 'biased'):
        model_cfg.biased = False
    if not hasattr(model_cfg, 'n_last_reg_layers'):
        model_cfg.n_last_reg_layers = -1
    if not hasattr(model_cfg, 'loo_cond_mean'):
        model_cfg.loo_cond_mean = False
    if not hasattr(model_cfg, 'zy_cov'):
        model_cfg.zy_cov = False
    if not hasattr(data_cfg, 'regress'):
        data_cfg.regress = True
        print('Regress_ZY set to True.')
    if not hasattr(model_cfg, 'regression'):
        model_cfg.regression = 'linear_old'
    if ('dsprites' in data_cfg.data_key) and not hasattr(data_cfg, 'holdout_ratio'):
        data_cfg.holdout_ratio = 0.01
    if ('dsprites' in data_cfg.data_key or 'yale' in data_cfg.data_key) \
            and not hasattr(data_cfg, 'use_holdout_to_train'):
        data_cfg.use_holdout_to_train = False

    return {"data_cfg": data_cfg, "model_cfg": model_cfg, "exp_cfg": exp_cfg}


def dir_util(cfg, seed, sweep):
    dirs = dict()
    dirs['output_location'] = join(cfg.output_location, splitext(basename(cfg.cfg_file))[0], str(seed))
    if sweep:
        dirs['output_location'] = join(dirs['output_location'], 'sweep')
    os.makedirs(dirs['output_location'], exist_ok=True)

    return dirs
