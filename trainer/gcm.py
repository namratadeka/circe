import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from sklearn import linear_model
from collections import defaultdict
from scipy.linalg import solve as scp_solve

from utils import utils, losses, wandb_utils
from trainer.base_trainer import BaseTrainer


class GCM(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        super().__init__(data_cfg, model_cfg, exp_cfg)
    
    def _set_kernels(self):
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
        if 'linear' in self.model_cfg.regression:
            for mode in ['train', 'train_ood']:
                try:
                    self.yz_reg['coef'] = torch.FloatTensor(self.dataloaders[mode].dataset.linear_reg.coef_).to(
                        self.device)
                    self.yz_reg['intercept'] = torch.FloatTensor(self.dataloaders[mode].dataset.linear_reg.intercept_).to(
                        self.device)
                except:
                    continue
        
        elif self.model_cfg.regression == 'kernel-ridge':
            # save memory
            for mode in self.model_cfg.modes:
                del self.dataloaders[mode].dataset.linear_reg
            for mode in ['train', 'train_ood']:
                try:
                    Y_heldout = torch.FloatTensor(self.dataloaders[mode].dataset.targets_heldout)
                    Z_heldout = torch.FloatTensor(self.dataloaders[mode].dataset.distractors_heldout)
                    n_points = Y_heldout.shape[0]

                    if self.model_cfg.loo_cond_mean:
                        Kz = torch.mm(Z_heldout, Z_heldout.T)
                        self.Y_heldout = Y_heldout
                        print('Estimating regressions parameters with LOO')
                        reg_list = [1e-2, 1e-1, 1.0, 10.0, 100.0]
                        sigma2_list = [1.0, 0.1, 0.01, 0.001]
                        self._leave_one_out_regressors(reg_list, sigma2_list, Kz)

                    Ky = eval(f'losses.{self.kernel_y}_kernel(Y_heldout, **self.kernel_y_args)')
                    I = torch.eye(n_points, device=Ky.device)
                    print('All gram matrices computed')
                    # self.yz_reg['W'] = torch.linalg.solve((Ky + self.model_cfg.ridge_lambda*I), Z_heldout).to(self.device)
                    self.yz_reg['W'] = torch.tensor(scp_solve(np.float128((Ky + self.model_cfg.ridge_lambda*I).cpu().numpy()),
                                                        np.float128(Z_heldout.cpu().numpy()),
                                                        assume_a='pos')).float().to(self.device)
                    self.yz_reg['Y'] = Y_heldout.to(self.device)
                    print('W computed')
                except:
                    continue

    def _batch_regress_yx(self, y, fx, y_prev, fx_prev):
        # for the first batch use the same data to regress on.
        if y_prev == None:
            y_prev = y.clone().detach()
        if fx_prev == None:
            fx_prev = fx.clone().detach()

        if self.model_cfg.regression == 'linear':
            y_prev = y_prev.cpu().numpy()
            fx_prev = fx_prev.clone().detach().cpu().numpy()
            linear_reg = linear_model.LinearRegression()
            linear_reg.fit(y_prev, fx_prev)

            fx_ = np.matmul(y.cpu().numpy(), linear_reg.coef_.T) + linear_reg.intercept_
            return torch.FloatTensor(fx_).to(self.device)

        elif self.model_cfg.regression == 'linear_old':
            y = y.cpu().numpy()
            fx = fx.clone().detach().cpu().numpy()
            linear_reg = linear_model.LinearRegression()
            linear_reg.fit(y, fx)

            fx_ = np.matmul(y, linear_reg.coef_.T) + linear_reg.intercept_
            return torch.FloatTensor(fx_).to(self.device)

        elif self.model_cfg.regression == 'kernel-ridge':
            KY = eval(f'losses.{self.kernel_y}_kernel(y_prev, **self.kernel_y_args)')
            I = torch.eye(KY.shape[0], device=KY.device)
            W = torch.linalg.solve((KY + self.model_cfg.ridge_lambda*I), fx_prev)

            KyY = eval(f'losses.{self.kernel_y}_kernel(y, Y=y_prev, **self.kernel_y_args)')
            fx_ = KyY @ W
            return fx_

    def _epoch(self, epochID, mode):
        '''
        Run a single epoch, aggregate losses & log to wandb.
        '''
        train = 'train' in mode
        self.model.train() if train else self.model.eval()

        all_losses = defaultdict(list)

        data_iter = iter(self.dataloaders[mode])
        tqdm_iter = tqdm(range(len(self.dataloaders[mode])), dynamic_ncols=True)

        y_prev = None
        fx_prev = None

        for i in tqdm_iter:
            batch = utils.dict_to_device(next(data_iter), self.device)
            x, y, z = batch['x'], batch['y'], batch['z']
            if train:
                ft, y_ = self.model(x)
            else:
                with torch.no_grad():
                    ft, y_ = self.model(x)

            # supervised target loss:
            if self.model_cfg.model_key == 'regressor':
                target_loss = F.mse_loss(y_, y)
            elif self.model_cfg.model_key == 'classifier':
                label = batch['label']
                label[label > self.model_cfg.target_threshold] = 1
                target_loss = F.nll_loss(y_, label)

            # cond. ind. test-statistic:
            fx = ft[-1]
            if 'linear' in self.model_cfg.regression:
                z_ = torch.mm(y, self.yz_reg['coef']) + self.yz_reg['intercept']
            elif self.model_cfg.regression == 'kernel-ridge':
                KyY = eval(f'losses.{self.kernel_y}_kernel(y, Y=self.yz_reg["Y"], **self.kernel_y_args)').to(self.device)
                z_ = KyY @ self.yz_reg['W']
            fx_ = self._batch_regress_yx(y, fx, y_prev, fx_prev)
            gcm = losses.gcm(fx, fx_, z, z_)
            loss = target_loss + self.model_cfg.lamda*gcm

            if train:
                self._backprop(loss)

            tqdm_iter.set_description("V: {} | Epoch: {} | {} | Loss: {:.4f}".format(
                self.exp_cfg.version, epochID, mode, loss.item()
            ), refresh=True)

            all_losses['target_loss'].append(target_loss.item())
            all_losses['gcm'].append(gcm.item())
            all_losses['total_loss'].append(loss.item())

            y_prev = y.clone().detach()
            fx_prev = fx.clone().detach()
        
        all_losses = utils.aggregate(all_losses)
        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_summary(epochID, mode, all_losses)

        return all_losses['total_loss']


class GCMTrainerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        if not self._instance:
            self._instance = GCM(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
