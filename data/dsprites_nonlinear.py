'''
dsprites torch dataset, non-linear dependence between Y and Z.
'''
import torch
import numpy as np

from data.dsprites import Dsprites

factor_map = {
    'color' : 0,
    'shape' : 1,
    'scale' : 2,
    'orientation' : 3,
    'position_x' : 4,
    'position_y' :  5
}

class DspritesNonLinear(Dsprites):
    def __init__(self, path: str, noise: float, distractor: str, target: str, ood: bool, nl_type: str,
                 regress:bool=True, holdout_ratio:float=0.01, use_holdout_to_train=False):
        self.nl_type = nl_type
        super().__init__(path, noise, distractor, target, ood, regress, holdout_ratio, use_holdout_to_train)

    def correlate_tricky(self, distractor: str, target: str):
        print("Sampling Z-Y correlated data.")
        N = self.latent_classes.shape[0]

        # y_ = y + noise
        # xi = N(0, 1)
        # z = y + xi
        # z_ = y + 0.1 xi^2

        if distractor == 'position_x' and target == 'position_y':
            y_pos = self.latent_classes[:, factor_map['position_y']] + 1
            x_pos = self.latent_classes[:, factor_map['position_x']] + 1
            print('y_pos min {} max {}'.format(y_pos.min(), y_pos.max()))
            print('x_pos min {} max {}'.format(x_pos.min(), x_pos.max()))
            indices = np.array([])
            noise_draws = np.array([])

            while len(indices) < N:
                for y in range(1, 33):
                    y_idx = np.where(y_pos == y)[0]
                    noise = 2 * (np.random.randint(0, 2, N) - 0.5) * np.sqrt(np.random.randint(0, 4, N))
                    x_idx = np.where(np.abs(x_pos - noise ** 2 - y) < 1)[0]
                    valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))), dtype=int)
                    indices = np.concatenate([indices, valid_idx])
                    noise_draws = np.concatenate([noise_draws, noise[valid_idx]])

            p = np.random.permutation(len(indices))
            indices = indices[p][:N].astype(np.int)
            noise_draws = noise_draws[p][:N]
            self.images = self.images[indices]
            self.latent_classes = self.latent_classes[indices]
            self.latent_values = self.latent_values[indices]
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
            self.distractors = torch.FloatTensor(noise_draws)

    def correlate(self, distractor: str, target: str):
        if self.nl_type == 'tricky':
            self.correlate_tricky(distractor, target)
        else:
            print("Sampling Z-Y correlated data.")
            N = self.latent_classes.shape[0]

            if distractor == 'position_x' and target == 'position_y':
                y_pos = self.latent_classes[:, factor_map['position_y']] + 1
                x_pos = self.latent_classes[:, factor_map['position_x']] + 1
                print('y_pos min {} max {}'.format(y_pos.min(), y_pos.max()))
                print('x_pos min {} max {}'.format(x_pos.min(), x_pos.max()))
                eps = np.abs(np.random.normal(loc=0, scale=1, size=N))
                indices = np.array([])

                for y in range(1, 33):
                    y_idx = np.where(y_pos == y)[0]
                    if self.nl_type == 'cone':
                        for tries in range(10):
                            eps = np.random.normal(loc=0, scale=7 * y / 32, size=1)
                            if eps > 0:
                                x1 = np.where(x_pos >= y)[0]
                                x2 = np.where(x_pos <= y + eps)[0]
                            else:
                                x1 = np.where(x_pos <= y)[0]
                                x2 = np.where(x_pos >= y + eps)[0]
                            x_idx = np.array(list(set(x1.tolist()).intersection(set(x2.tolist()))), dtype=int)
                            valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))), dtype=int)
                            indices = np.concatenate([indices, valid_idx])
                    elif self.nl_type == 'y-cone':
                        for tries in range(10):
                            eps = 0.5 * 2 * (np.random.randint(0, 2, size=1) - 0.5) * y ** 2 / 32
                            x_idx = np.where(np.abs(x_pos - 0.5 * y - eps) < 1)[0]
                            valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))), dtype=int)
                            indices = np.concatenate([indices, valid_idx])
                    else:
                        if self.nl_type == 'quadratic':
                            x1 = np.where(x_pos >= (y_pos)**2 / 32 - eps)[0]
                            x2 = np.where(x_pos <= (y_pos)**2 / 32 + eps)[0]
                        elif self.nl_type == 'quadratic-centered':
                            x1 = np.where(x_pos/32 >= 4*(y_pos/32 - 0.5)**2 - eps/64)[0]
                            x2 = np.where(x_pos/32 <= 4*(y_pos/32 - 0.5)**2 + eps/64)[0]
                        x_idx = np.array(list(set(x1.tolist()).intersection(set(x2.tolist()))))
                        valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))))
                        indices = np.concatenate([indices, valid_idx])

                np.random.shuffle(indices)
                indices = np.random.choice(indices, N, replace=True)
                indices = np.asarray(indices, dtype=int)
                self.images = self.images[indices]
                self.latent_classes = self.latent_classes[indices]
                self.latent_values = self.latent_values[indices]
                self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
                self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31

class DspritesNonLinearBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, path:str, noise:float, distractor:str, target:str, ood:bool, nl_type:str,
                 regress:bool, holdout_ratio:float, use_holdout_to_train:bool, **_ignored):
        self._instance = DspritesNonLinear(path=path, noise=noise, distractor=distractor,
                                  target=target, ood=ood, nl_type=nl_type, regress=regress, holdout_ratio=holdout_ratio,
                                           use_holdout_to_train=use_holdout_to_train)
        return self._instance
