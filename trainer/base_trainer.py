import sys
import wandb
import torch
import numpy as np
import torch.optim as optim
from abc import abstractmethod, ABC

from data import data
from model import model
from utils import utils


class BaseTrainer(ABC):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataloaders = data.create_dataloaders(data_cfg, model_cfg.modes)

        self._set_kernels()  # goes first to correctly setup kernels
        self._get_yz_regressors()
        self._setup_model()
        self._setup_optimizers()
        self._setup_schedulers()

        self.last_best = -1
        self.val_loss = np.inf

    @abstractmethod
    def _get_yz_regressors(self):
        pass

    @abstractmethod
    def _set_kernels(self):
        pass

    def _setup_model(self):
        print("Initializing networks.")
        self.model = model.factory.create(self.model_cfg.model_key, **{"model_cfg": self.model_cfg}).to(self.device)
        if self.exp_cfg.load is not None:
            saved_model = torch.load(self.exp_cfg.load, map_location=self.device)
            utils.copy_state_dict(self.model.state_dict(), saved_model['model'])
        if self.exp_cfg.wandb:
            wandb.watch(self.model)

    def _setup_optimizers(self):
        print("Initializing optimizers.")
        params = list(self.model.parameters())
        optimizer = self.model_cfg.optimizer

        self.opt = eval("optim.{}(params, **{})".format([*optimizer.keys()][0],
                                                        [*optimizer.values()][0]))
        if self.exp_cfg.resume:
            saved_opt = torch.load(self.exp_cfg.load, map_location=self.device)['optimizer']
            self.opt.load_state_dict(saved_opt)

    def _setup_schedulers(self):
        scheduler = self.model_cfg.scheduler
        self.scheduler = eval("optim.lr_scheduler.{}(self.opt, **{})".format([*scheduler.keys()][0],
                                                                             [*scheduler.values()][0]))

    def _backprop(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    @abstractmethod
    def _epoch(self, epochID, mode):
        '''
        Run a single epoch, aggregate losses & log to wandb.
        '''
        pass

    def save(self, epochID, loss):
        '''
        Save on improvement as well as every 5 epochs.
        Early stopping.
        '''
        save = False
        if loss < self.val_loss:
            self.val_loss = loss
            save = True
            self.last_best = epochID
            ckpt_type = 'best'
        elif epochID - self.last_best > self.model_cfg.patience:
            sys.exit(f"No improvement in the last {self.model_cfg.patience} epochs. EARLY STOPPING.")
        elif epochID > 0 and epochID % 5 == 0:
            save = True
            ckpt_type = 'latest'
        if save:
            utils.save_model(self.model, self.opt, epochID, loss, self.exp_cfg.output_location, ckpt_type)

    def run(self):
        '''
        Run training/inference and save checkpoints.
        '''
        print("Beginning run:")
        for epoch in range(self.model_cfg.epochs):
            for mode in self.model_cfg.modes:
                loss = self._epoch(epoch, mode)
                if 'train' in mode:
                    self.scheduler.step()
                if mode == 'val':
                    self.save(epoch, loss)
                elif mode == 'val_ood' and 'val' not in self.model_cfg.modes:
                    self.save(epoch, loss)


class BaseTrainerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        if not self._instance:
            self._instance = BaseTrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
