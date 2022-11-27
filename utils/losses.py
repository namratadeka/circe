'''
Define custom losses and test-statistics here.
'''
import torch
import numpy as np


def compute_pdist_sq(x, y=None):
    """compute the squared paired distance between x and y."""
    if y is not None:
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0)
    a = x.view(x.shape[0], -1)
    aTa = torch.mm(a, a.T)
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)

def gaussian_kernel(X, sigma2=1.0, Y=None, normalized=False, **ignored):
    if normalized:
        X = X / torch.linalg.norm(X, dim=1, keepdim=True)
        if Y is not None:
            Y = Y / torch.linalg.norm(Y, dim=1, keepdim=True)
    Dxx = compute_pdist_sq(X, Y)
    if sigma2 is None:
        sigma2 = Dxx.median()
    Kx = torch.exp(-Dxx / sigma2)
    return Kx

def hsic_matrices(Kx, Ky, biased=False):
    n = Kx.shape[0]

    if biased:
        a_vec = Kx.mean(dim=0)
        b_vec = Ky.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        return (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

    else:
        tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
        tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

        u = tilde_Kx * tilde_Ky
        k_row = tilde_Kx.sum(dim=1)
        l_row = tilde_Ky.sum(dim=1)
        mean_term_1 = u.sum()  # tr(KL)
        mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
        mu_x = tilde_Kx.sum()
        mu_y = tilde_Ky.sum()
        mean_term_3 = mu_x * mu_y

        # Unbiased HISC.
        mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean

def hsic(X, Y, kernelX='gaussian', kernelX_params=None, kernelY='linear', kernelY_params=None, biased=False):
    '''X ind. Y'''
    # todo:
    #  alternative implementation for RFF
    #  biased/unbiased HSIC choice
    #  faster implementation for biased
    Kx = eval(f'{kernelX}_kernel(X, **kernelX_params)')
    Ky = eval(f'{kernelY}_kernel(Y, **kernelY_params)')

    return hsic_matrices(Kx, Ky, biased)

def hscic(X, Z, Y, ridge_lambda, kernelX='gaussian', kernelX_params=None,
          kernelZ='gaussian', kernelZ_params=None, kernelY='gaussian', kernelY_params=None):
    '''X ind. Z | Y '''
    # todo:
    #  alternative implementation for RFF
    Kx = eval(f'{kernelX}_kernel(X, **kernelX_params)')
    Kz = eval(f'{kernelZ}_kernel(Z, **kernelZ_params)')
    Ky = eval(f'{kernelY}_kernel(Y, **kernelY_params)')

    # https://arxiv.org/pdf/2207.09768.pdf
    WtKyy = torch.linalg.solve(Ky + ridge_lambda  * torch.eye(Ky.shape[0]).to(Ky.device), Ky) # * Ky.shape[0] for ridge_lambda
    # todo:
    #   SVD + LOO here? but that doesn't scale with # params
    #   three LOO for all three regressions?
    #   re-do it every few iters?
    # sum_i A_(i.)B_(.i) = tr(AB) = (A * B^T).sum()
    # A = Kyy^T, B = the other one, so the transposes cancel out
    term_1 = (WtKyy * ((Kx * Kz) @ WtKyy)).sum() # tr(WtKyy.T @ (Kx * Kz) @ WtKyy)
    WkKxWk = WtKyy * (Kx @ WtKyy)
    KzWk = Kz @ WtKyy
    term_2 = (WkKxWk * KzWk).sum()
    # here it's crucial that the first dimension is the batch of other matrices
    term_3 = (WkKxWk.sum(dim=0) * (WtKyy * KzWk).sum(dim=0)).sum()

    return (term_1 - 2 * term_2 + term_3) / Ky.shape[0]

def circe_estimate(X, Z, Z_heldout, Y, Y_heldout, W_1, W_2, kernelX='gaussian', kernelX_params=None,
                   kernelZ='gaussian', kernelZ_params=None, kernelY='gaussian', kernelY_params=None,
                   biased=False, cond_cov=False):
    '''X ind. Z | Y '''
    # todo:
    #  alternative implementation for RFF

    Z_all = torch.vstack((Z, Z_heldout))
    Kz_all = eval(f'{kernelZ}_kernel(Z_all, Y=Z, **kernelZ_params)')  # n_all x n_batch

    Y_all = torch.vstack((Y, Y_heldout))
    Ky_all = eval(f'{kernelY}_kernel(Y_all, Y=Y, **kernelY_params)')  # n_heldout x n_batch

    n_points = Y.shape[0]

    A = (0.5 * Ky_all[n_points:, :].T @ W_2 - Kz_all[n_points:, :].T) @ W_1 @ Ky_all[n_points:, :]
    Kres = Kz_all[:n_points, :n_points] + A + A.T
    Kres = Kres * Ky_all[:n_points, :]

    Kx = eval(f'{kernelX}_kernel(X, **kernelX_params)')

    if cond_cov:
        Kx = Kx * Kres
        if biased:
            return Kx.mean()
        idx = torch.triu_indices(n_points, n_points, 1)
        return Kx[idx[0], idx[1]].mean()
    return hsic_matrices(Kx, Kres, biased)

def gcm(x, fz, y, gz):
    '''
    Generalized covariance measure for multivariate X & Y.
    From https://arxiv.org/abs/1804.07203, Eq.(3)
    '''
    n = x.shape[0]

    residual_x = x - fz
    residual_y = y - gz
    R = torch.bmm(residual_x.unsqueeze(-1), residual_y.unsqueeze(1))
    R_avg = R.mean(dim=0)
    tau_N = np.sqrt(n) * R_avg
    tau_D = torch.sqrt((R ** 2).mean(dim=0) - (R_avg ** 2))
    T_n = torch.div(tau_N, tau_D + 1e-10)

    S_n = torch.abs(T_n).max()
    return S_n
