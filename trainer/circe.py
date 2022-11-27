import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from scipy.linalg import solve as scp_solve

from trainer.base_trainer import BaseTrainer
from utils import utils, losses, wandb_utils


class CIRCE(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        super().__init__(data_cfg, model_cfg, exp_cfg)

    def _set_kernels(self):
        self.kernel_ft = [*self.model_cfg.kernel_ft.keys()][0]
        self.kernel_ft_args = [*self.model_cfg.kernel_ft.values()][0]
        self.kernel_z = [*self.model_cfg.kernel_z.keys()][0]
        self.kernel_z_args = [*self.model_cfg.kernel_z.values()][0]
        self.kernel_y = [*self.model_cfg.kernel_y.keys()][0]
        self.kernel_y_args = [*self.model_cfg.kernel_y.values()][0]

    def _leave_one_out_regressors(self, reg_list, sigma2_list, Kz):
        LOO_error_sanity_check = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_error = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_tol = np.zeros(len(sigma2_list))
        for idx, sigma2 in enumerate(sigma2_list):
            print(idx, sigma2)
            self.kernel_y_args['sigma2'] = sigma2
            loo_size = self.Y_heldout.shape[0] // 2
            K_YY = eval(f'losses.{self.kernel_y}_kernel(self.Y_heldout, **self.kernel_y_args)')
            LOO_error_sanity_check[idx] = utils.leave_one_out_reg(K_YY[:loo_size, :loo_size].cpu().numpy(),
                                                            labels=Kz[:loo_size, loo_size:].cpu().numpy(),
                                                            reg_list=reg_list)
            LOO_error[idx], under_tol, LOO_tol[idx] = \
                utils.leave_one_out_reg_kernels(K_YY.cpu().numpy(), Kz.cpu().numpy(), reg_list)
            # if not np.any(under_tol)
            LOO_error[idx, under_tol] = 2.0 * LOO_error[idx].max()  # a hack to remove < tol lambdas
        LOO_idx = np.unravel_index(np.argmin(LOO_error, axis=None), LOO_error.shape)
        self.kernel_y_args['sigma2'] = sigma2_list[LOO_idx[0]]
        self.model_cfg.ridge_lambda = reg_list[LOO_idx[1]]
        print('Best LOO parameters: sigma2 {}, lambda {}'.format(sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))
        if self.model_cfg.ridge_lambda < LOO_tol[LOO_idx[0]]:
            print('POORLY CONDITIONED MATRIX, switching lambda to SVD tolerance: {}'.format(LOO_tol[LOO_idx[0]]))
            self.model_cfg.ridge_lambda = LOO_tol[LOO_idx[0]]

        LOO_idx = np.unravel_index(np.argmin(LOO_error_sanity_check, axis=None), LOO_error_sanity_check.shape)
        print('Best LOO parameters (sanity check): sigma2 {}, lambda {}'.format(
            sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))

        print('LOO results\n{}'.format(LOO_error))
        print('LOO results (sanity check)\n{}'.format(LOO_error_sanity_check))

    def _get_yz_regressors(self):
        self.yz_reg = defaultdict(dict)

        print('YZ regressors correction')
        # save memory
        for mode in self.model_cfg.modes:
            del self.dataloaders[mode].dataset.linear_reg

        train_modes = ['train', 'train_ood']
        # todo: add other modes
        print('ONLY TRAIN/VAL HSIC IS CORRECT, OOD IS NOT')
        for mode in train_modes:
            try:
                self.Y_heldout = torch.FloatTensor(self.dataloaders[mode].dataset.targets_heldout)
                self.Z_heldout = torch.FloatTensor(self.dataloaders[mode].dataset.distractors_heldout)

                print('Points saved')
                n_points = self.Y_heldout.shape[0]
                Kz = eval(f'losses.{self.kernel_z}_kernel(self.Z_heldout, **self.kernel_z_args)')

                if self.model_cfg.loo_cond_mean:
                    print('Estimating regressions parameters with LOO')
                    reg_list = [1e-2, 1e-1, 1.0, 10.0, 100.0]
                    sigma2_list = [1.0, 0.1, 0.01, 0.001]
                    self._leave_one_out_regressors(reg_list, sigma2_list, Kz)

                Ky = eval(f'losses.{self.kernel_y}_kernel(self.Y_heldout, **self.kernel_y_args)')
                I = torch.eye(n_points, device=Ky.device)
                print('All gram matrices computed')
                W_all = torch.tensor(scp_solve(np.float128((Ky + self.model_cfg.ridge_lambda * I).cpu().numpy()),
                                                    np.float128(torch.cat((I, Kz), 1).cpu().numpy()),
                                                    assume_a='pos')).float().to(Ky.device)
                print('W_all computed')
                self.W_1 = W_all[:, :n_points].to(self.device)
                self.W_2 = W_all[:, n_points:].to(self.device)

                self.Y_heldout = self.Y_heldout.to(self.device)
                self.Z_heldout = self.Z_heldout.to(self.device)
            except:
                continue

    def _epoch(self, epochID, mode):
        '''
        Run a single epoch, aggregate losses & log to wandb.
        '''
        train = 'train' in mode
        self.model.train() if train else self.model.eval()

        all_losses = defaultdict(list)

        data_iter = iter(self.dataloaders[mode])
        tqdm_iter = tqdm(range(len(self.dataloaders[mode])), dynamic_ncols=True)

        for i in tqdm_iter:
            batch = utils.dict_to_device(next(data_iter), self.device)
            x, y, z = batch['x'], batch['y'], batch['z']
            if train:
                ft, y_ = self.model(x)
            else:
                with torch.no_grad():
                    ft, y_ = self.model(x)

            # if self.model_cfg.zy_cov:
            #     z = torch.hstack((z, y))

            # supervised target loss:
            if self.model_cfg.model_key == 'regressor':
                target_loss = F.mse_loss(y_, y)
            elif self.model_cfg.model_key == 'classifier':
                label = batch['label']
                label[label > self.model_cfg.target_threshold] = 1
                target_loss = F.nll_loss(y_, label)
            
            # CIRCE regularizer:
            circe = 0
            if self.model_cfg.n_last_reg_layers == -1 or self.model_cfg.n_last_reg_layers > len(ft):
                self.model_cfg.n_last_reg_layers = len(ft)
            for int_ft in ft[-self.model_cfg.n_last_reg_layers:]:
                circe += losses.circe_estimate(int_ft, z, self.Z_heldout, y, self.Y_heldout, self.W_1, self.W_2,
                                              self.kernel_ft, self.kernel_ft_args,
                                              self.kernel_z, self.kernel_z_args, self.kernel_y, self.kernel_y_args,
                                              self.model_cfg.biased, cond_cov=not self.model_cfg.centered_circe)
            loss = target_loss + self.model_cfg.lamda * circe

            if train:
                self._backprop(loss)

            tqdm_iter.set_description("V: {} | Epoch: {} | {} | Loss: {:.4f}".format(
                self.exp_cfg.version, epochID, mode, loss.item()
            ), refresh=True)

            all_losses['target_loss'].append(target_loss.item())
            all_losses['circe_c'].append(circe.item())
            all_losses['total_loss'].append(loss.item())

        all_losses = utils.aggregate(all_losses)
        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_summary(epochID, mode, all_losses)

        return all_losses['total_loss']


class CIRCETrainerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        if not self._instance:
            self._instance = CIRCE(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
