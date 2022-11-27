import os
import torch
import random
import numpy as np


def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_model(model, opt, epoch, loss, outpath, ckpt_type):
    save_dict = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(save_dict, os.path.join(outpath, f'{ckpt_type}.pth'))

def copy_state_dict(cur_state_dict, pre_state_dict, prefix=""):
    """
        Load parameters
    Args:
        cur_state_dict (dict): current parameters
        pre_state_dict ([type]): load parameters
        prefix (str, optional): specific module names. Defaults to "".
    """

    def _get_params(key):
        key = prefix + key
        try:
            out = pre_state_dict[key]
        except Exception:
            try:
                out = pre_state_dict[key[7:]]
            except Exception:
                try:
                    out = pre_state_dict["module." + key]
                except Exception:
                    try:
                        out = pre_state_dict[key[14:]]
                    except Exception:
                        out = None
        return out

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print("parameter {} not found".format(k))
                # logging.info("parameter {} not found".format(k))
                continue
            cur_state_dict[k].copy_(v)
        except Exception:
            print("copy param {} failed".format(k))
            # logging.info("copy param {} failed".format(k))
            continue

def dict_to_device(d_ten: dict, device):
    """
    Sets a dictionary to device
    Args:
        d_ten (dict): dictionary of tensors
        device (str): torch device
    Returns:
        dict: dictionary on device
    """
    for key, tensor in d_ten.items():
        if type(tensor) is torch.Tensor:
            d_ten[key] = d_ten[key].to(device)

    return d_ten

def aggregate(metrics):
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    return metrics

def leave_one_out_reg_kernels_one(K_YY, K_QQ, reg):
    Kinv = np.linalg.solve(K_YY + reg * np.eye(K_YY.shape[0]), K_YY).T
    diag_idx = np.arange(K_YY.shape[0])
    return ((K_QQ[diag_idx, diag_idx] + (Kinv @ K_QQ @ Kinv)[diag_idx, diag_idx] -
             2 * (Kinv @ K_QQ)[diag_idx, diag_idx]) / (1 - Kinv[diag_idx, diag_idx]) ** 2).mean()

def leave_one_out_reg_kernels(K_YY, K_QQ, reg_list):
    loos = []
    for reg in reg_list:
        loos.append(leave_one_out_reg_kernels_one(K_YY, K_QQ, reg))
    U, eigs = np.linalg.svd(K_YY, hermitian=True)[:2]
    svd_tol = eigs.max() * U.shape[0] * np.finfo(U.dtype).eps
    regs = np.array(reg_list)
    return loos, regs < svd_tol, svd_tol

def leave_one_out_reg(K_YY, labels, reg_list):
    U, eigs = np.linalg.svd(K_YY, hermitian=True)[:2]
    regs = np.array(reg_list)
    eigs_reg_inv = 1 / (eigs[:, None] + regs[None, :]) # rank x n_regs

    KU = U * eigs[None, :]#K_YY @ U
    Ul = U.T @ labels
    preds = np.tensordot(KU, Ul[:, :, None] * eigs_reg_inv[:, None, :], axes=1) # rank x label_dim x n_regs
    # A = (U * eigs / (eigs + reg)) @ U.T
    A_ii = (U ** 2 @ (eigs[:, None] * eigs_reg_inv))  # rank x n_regs

    return np.mean(((labels[:, :, None] - preds) / (1 - A_ii)[:, None, :]) ** 2, axis=(0, 1))
