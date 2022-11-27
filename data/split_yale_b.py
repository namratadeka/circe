'''
Split Extended Yale-B data into train-val-test sets.
'''
import os
import shutil
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split

path = './ExtendedYaleB'
ids = np.array(os.listdir(path))

print("Splitting Extended Yale-B.")
train, val = train_test_split(range(len(ids)), test_size=0.2, random_state=42)
val, test = train_test_split(val, test_size=0.5, random_state=42)

os.makedirs(join(path, 'train'), exist_ok=True)
for folder in ids[train]:
    shutil.move(join(path, folder), join(path, 'train'))

os.makedirs(join(path, 'val'), exist_ok=True)
for folder in ids[val]:
    shutil.move(join(path, folder), join(path, 'val'))

os.makedirs(join(path, 'test'), exist_ok=True)
for folder in ids[test]:
    shutil.move(join(path, folder), join(path, 'test'))
