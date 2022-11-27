'''
Yale B Extended torch dataset
'''
from collections import defaultdict
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from os.path import join, basename, splitext, dirname

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class YaleBExtended(Dataset):
    def __init__(self, path:str, noise:float, ood:bool, nl_type:str, holdout_ratio:float, use_holdout_to_train=False):
        ids = os.listdir(path)
        self.images = []
        self.holdout_ratio = holdout_ratio
        self.use_holdout_to_train = use_holdout_to_train
        for id in ids:
            self.images.extend(glob(join(path, id, '*E*.pgm')))

        self._set_id_info(ids)
        self.all_images = np.array(self.images).copy()

        labels = [splitext(basename(x))[0].split('_')[1] for x in self.images]
        self.targets = np.array([int(x[1:3]) for x in labels]) # pose
        self.distractors = np.array([int(x[4:8]) for x in labels], dtype=float) # azimuth
        self.elevation = np.array([int(x[9:]) for x in labels], dtype=float)

        self.nl_type=nl_type
        if not ood:
            self.correlate()

        self.set_noise(path, noise)
        self.targets = self.targets / 9
        self.distractors = self.distractors / 130

        self._regress_YZ()

        transforms = [
            T.Normalize(mean=(0.5), std=(0.5)),
            T.Resize((224, 224))
        ]
        self.transforms = T.Compose(transforms)

    def _regress_YZ(self):
        '''
        Create a held-out set and learn a linear regressor from Y to Z on it.
        '''
        print("Computing Y->Z residuals.")

        train, test = train_test_split(range(len(self.targets)), test_size=1 - self.holdout_ratio, random_state=42)

        Y = self.targets[train].reshape(-1, 1)
        Z = self.distractors[train].reshape(-1, 1)
        self.linear_reg = linear_model.LinearRegression()
        self.linear_reg.fit(Y, Z)

        self.images = np.array(self.images)

        if self.use_holdout_to_train:
            print('\n\nHOLDOUT DATA WILL BE USED FOR TRAINING\n\n')
        else:
            self.targets = self.targets[test]
            self.distractors = self.distractors[test]
            self.images = self.images[test]

        self.targets_heldout = Y
        self.distractors_heldout = Z

        print('Train size: {}, Heldout size: {}'.format(self.targets.shape[0], Y.shape[0]))

    def _set_id_info(self, ids):
        self.id_info = {}
        for id in ids:
            self.id_info[id] = defaultdict(list)
        for i in range(self.__len__()):
            label = splitext(basename(self.images[i]))[0]
            id = label.split('_')[0]
            pose = int(label.split('_')[1][1:3])
            azimuth = int(label.split('_')[1][4:8])
            elevation = int(label.split('_')[1][9:])
            if pose not in self.id_info[id]['pose']:
                self.id_info[id]['pose'].append(pose)
            if azimuth not in self.id_info[id]['azimuth']:
                self.id_info[id]['azimuth'].append(azimuth)
            if elevation not in self.id_info[id]['elevation']:
                self.id_info[id]['elevation'].append(elevation)
        for id in ids:
            self.id_info[id]['pose'] = sorted(self.id_info[id]['pose'])
            self.id_info[id]['azimuth'] = sorted(self.id_info[id]['azimuth'])
            self.id_info[id]['elevation'] = sorted(self.id_info[id]['pose'])

    def set_noise(self, path, noise):
        poses = list(range(10))
        sorted_images = sorted(self.all_images)
        shift = np.random.normal(loc=0, scale=noise, size=self.__len__())
        for i in tqdm(range(self.__len__())):
            pose = self.targets[i]
            azimuth = int(self.distractors[i])
            az_sign = '+' if azimuth >= 0 else '-'
            elevation = int(self.elevation[i])
            el_sign = '+' if elevation >= 0 else '-'
            id = basename(dirname(self.images[i]))
            pose_idx = poses.index(pose)
            noisy_pose = int(pose_idx + shift[i])

            if noisy_pose < len(poses) and noisy_pose >= 0 and poses[noisy_pose] in self.id_info[id]['pose']:
                new_pose = poses[noisy_pose]
            else:
                new_pose = pose

            fname = '{}/{}_P{:02d}A{}{:03d}E{}{:02d}.pgm'.format(id, id, new_pose, az_sign, abs(azimuth), el_sign, abs(elevation))
            try:
                self.images[i] = sorted_images[sorted_images.index(join(path, fname))]
            except:
                print('noisy image not found')
                import pdb; pdb.set_trace()

    def correlate(self):
        indices = np.array([])
        unique_poses = np.unique(self.targets)

        while len(indices) < self.__len__():
            eps = np.abs(np.random.normal(loc=0, scale=1, size=self.__len__()))
            for y in unique_poses:
                y_idx = np.where(self.targets == y)[0]
                if self.nl_type == 'mod':
                    z1 = np.where(self.targets >= 0.5*np.abs(self.distractors/10) - eps/2)[0]
                    z2 = np.where(self.targets <= 0.5*np.abs(self.distractors/10) + eps/2)[0]
                    z_idx = np.array(list(set(z1.tolist()).intersection(set(z2.tolist()))))

                    valid_idx = np.array(list(set(z_idx.tolist()).intersection(set(y_idx.tolist()))))
                    indices = np.concatenate([indices, valid_idx])
                elif self.nl_type == 'y-cone':
                    az = 0.5*(self.distractors/130 + 1) * 9
                    for _ in range(10):
                        eps = 0.5 * 2 * (np.random.randint(0, 2, size=1) - 0.5) * y ** 2 / 9
                        z_idx = np.where(np.abs(az - 0.5 * y - eps) < 1)[0]
                        valid_idx = np.array(list(set(z_idx.tolist()).intersection(set(y_idx.tolist()))))
                        indices = np.concatenate([indices, valid_idx])
        p = np.random.permutation(len(indices))
        indices = indices[p][:self.__len__()].astype(np.int)
        self.images = np.array(self.images)[indices]
        self.targets = self.targets[indices]
        self.distractors = self.distractors[indices]
        self.elevation = self.elevation[indices]

    def __len__(self):
        return len(self.images)

    def _read_image(self, index):
        file = open(self.images[index], 'rb')
        img = np.expand_dims(plt.imread(file), -1).transpose(2, 0, 1) / 255.
        img = np.repeat(img, 3, 0)
        img = self.transforms(torch.FloatTensor(img))
        return img

    def __getitem__(self, index):
        image = self._read_image(index)

        return {
            'x': image,
            'y': torch.FloatTensor(self.targets[index : index + 1]),
            'z': torch.FloatTensor(self.distractors[index : index + 1])
        }

class YaleBExtendedBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, path:str, noise:float, ood:bool, nl_type:str, holdout_ratio:float, use_holdout_to_train:bool,
                 **_ignored):
        self._instance = YaleBExtended(path=path, noise=noise, ood=ood, nl_type=nl_type, holdout_ratio=holdout_ratio,
                                       use_holdout_to_train=use_holdout_to_train)
        return self._instance
