'''
dsprites torch dataset
'''
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms.functional import affine

factor_map = {
    'color' : 0,
    'shape' : 1,
    'scale' : 2,
    'orientation' : 3,
    'position_x' : 4,
    'position_y' :  5
}


class Dsprites(Dataset):
    def __init__(self, path:str, noise:float, distractor:str, target:str, ood:bool=False, regress:bool=True,
                 holdout_ratio:float=0.01, use_holdout_to_train=False):
        data = np.load(path)
        self.images = np.asarray(data['imgs'], dtype=np.float32)
        self.latent_values = data['latents_values']
        self.latent_classes = data['latents_classes']
        self.holdout_ratio = holdout_ratio
        self.use_holdout_to_train = use_holdout_to_train
        print('Holdout ration: {}'.format(holdout_ratio))

        if not ood:
            self.correlate(distractor, target)
        else:
            self.sample_ood(distractor, target)
        if regress:
            self._regress_YZ()
        else:
            print('Regression is NOT done!')
        
        self.set_noise(target, noise)
        self.normalize = Normalize(mean=(0.5), std=(0.5))

    def _regress_YZ(self):
        '''
        Create a held-out set and learn a linear regressor from Y to Z on it.
        '''
        print("Computing Y->Z residuals.")

        train, test = train_test_split(range(len(self.targets)), test_size=1 - self.holdout_ratio, random_state=42)

        Y = self.targets[train].numpy().reshape(-1, 1)
        Z = self.distractors[train].numpy().reshape(-1, 1)
        self.linear_reg = linear_model.LinearRegression()
        self.linear_reg.fit(Y, Z)

        if self.use_holdout_to_train:
            print('\n\nHOLDOUT DATA WILL BE USED FOR TRAINING\n\n')
        else:
            self.targets = self.targets[test]
            self.distractors = self.distractors[test]
            self.images = self.images[test]

        self.targets_heldout = Y
        self.distractors_heldout = Z

        print('Train size: {}, Heldout size: {}'.format(self.targets.shape[0], Y.shape[0]))

    def __len__(self):
        return self.images.shape[0]

    def set_noise(self, target:str, noise:int):
        N = len(self.images)
        shift = 2*np.random.normal(loc=0, scale=noise, size=N)
        self.translate = np.zeros((N, 2))
        if target == 'position_x':
            self.translate[:, 0] = shift
        elif target == 'position_y':
            self.translate[:, 1] = shift

    def sample_ood(self, distractor:str, target:str):
        print("Sampling OOD data.")
        if distractor == 'shape' and target == 'position_x':
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['shape']]) / 2
        elif distractor == 'position_x' and target == 'position_y':
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
            self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
        elif distractor == 'color' and target == 'position_x':
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            colors = np.linspace(0, 1, 6)[1:]
            colors = np.random.choice(colors, len(self.images))
            self.images *= np.expand_dims(np.expand_dims(colors, -1), -1)
            self.distractors = torch.FloatTensor(self.images.reshape(len(self.images), -1).max(axis=1))

    def correlate(self, distractor:str, target:str):
        print("Sampling Z-Y correlated data.")
        N = self.latent_classes.shape[0]
        if distractor == 'shape' and target == 'position_x':
            ns = 3
            corr = 1
            z_x = {
                0: 0,
                1: 10,
                2: 20,
                3: 32
            }
            indices = np.array([])
            shapes = self.latent_classes[:, factor_map['shape']] + 1
            x_pos = self.latent_classes[:, factor_map['position_x']] + 1
            for s in range(1, ns+1):
                s_idx = np.where(shapes == s)[0]
                valid1 = np.where(x_pos <= z_x[s])[0]
                valid2 = np.where(x_pos > z_x[s-1])[0]
                x_idx = np.array(list(set(valid1.tolist()).intersection(set(valid2.tolist()))))
                valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(s_idx.tolist()))))
                random_idx = np.random.choice(s_idx, int((1-corr)*valid_idx.shape[0]/corr), replace=False)
                indices = np.concatenate([indices, valid_idx, random_idx])
            np.random.shuffle(indices)
            indices = np.random.choice(indices, N, replace=True)
            indices = np.asarray(indices, dtype=int)

            self.images = self.images[indices]
            self.latent_classes = self.latent_classes[indices]
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['shape']]) / 2

        elif distractor == 'color' and target == 'position_x':
            nc = 5
            corr = 1
            z_x = {
                0: 0,
                1: 6,
                2: 12,
                3: 18,
                4: 24,
                5: 32
            }
            colors = np.linspace(0, 1, 6)[1:]
            x_pos = self.latent_classes[:, factor_map['position_x']] + 1
            self.targets = []
            self.distractors = []
            for c in range(1, nc+1):
                valid1 = np.where(x_pos <= z_x[c])[0]
                valid2 = np.where(x_pos > z_x[c-1])[0]
                x_idx = np.array(list(set(valid1.tolist()).intersection(set(valid2.tolist()))))
                x_idx = np.random.choice(x_idx, int(corr*x_idx.shape[0]), replace=False)
                self.images[x_idx] *= colors[c - 1]
            self.distractors = torch.FloatTensor(self.images.reshape(N, -1).max(axis=1))
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31

        elif distractor == 'position_x' and target == 'position_y':
            y_pos = self.latent_classes[:, factor_map['position_y']] + 1
            x_pos = self.latent_classes[:, factor_map['position_x']] + 1
            eps = np.random.normal(loc=0, scale=1, size=N)
            indices = np.array([])
            for x in range(1, 33):
                x_idx = np.where(x_pos == x)[0]
                y1 = np.where(y_pos >= x_pos - eps)[0]
                y2 = np.where(y_pos <= x_pos + eps)[0]
                y_idx = np.array(list(set(y1.tolist()).intersection(set(y2.tolist()))))
                valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))))
                indices = np.concatenate([indices, valid_idx])

            np.random.shuffle(indices)
            indices = np.random.choice(indices, N, replace=True)
            indices = np.asarray(indices, dtype=int)
            self.images = self.images[indices]
            self.latent_classes = self.latent_classes[indices]
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
            self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            
    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], -1).transpose(2, 0, 1)
        image = self.normalize(torch.FloatTensor(image))
        image = affine(image, angle=0, translate=tuple(self.translate[index]),
                         scale=1, shear=0, fill=-1.)

        return {
            'x': image,
            'y': self.targets[index : index + 1],
            'z': self.distractors[index: index + 1]
        }


class DspritesBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, path:str, noise:float, distractor:str, target:str, ood:bool, regress:bool,
                 holdout_ratio:float, use_holdout_to_train:bool,**_ignored):
        self._instance = Dsprites(path=path, noise=noise, distractor=distractor,
                                  target=target, ood=ood, regress=regress, holdout_ratio=holdout_ratio,
                                      use_holdout_to_train=use_holdout_to_train)
        return self._instance
