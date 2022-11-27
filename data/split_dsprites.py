'''
Split dsprites data into train-val-test sets.
'''
import numpy as np
from os.path import join, dirname
from sklearn.model_selection import train_test_split


path = './dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
data = np.load(path, encoding='latin1', allow_pickle=True)
images = data['imgs']
classes = data['latents_classes']
values = data['latents_values']

print("Splitting dsprites.")
N = classes.shape[0]
train, test = train_test_split(list(range(N)), test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)
train, val ,test = np.array(train), np.array(val), np.array(test)

train_images = images[train]
train_values = values[train]
train_classes = classes[train]
np.savez_compressed(join(dirname(path), 'dsprites_train'),
                    imgs=train_images,
                    latents_classes=train_classes,
                    latents_values=train_values)

val_images = images[val]
val_values = values[val]
val_classes = classes[val]
np.savez_compressed(join(dirname(path), 'dsprites_val'),
                    imgs=val_images,
                    latents_classes=val_classes,
                    latents_values=val_values)

test_images = images[test]
test_values = values[test]
test_classes = classes[test]
np.savez_compressed(join(dirname(path), 'dsprites_test'),
                    imgs=test_images,
                    latents_classes=test_classes,
                    latents_values=test_values)
