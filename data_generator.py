import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class DataGenerator(tf.keras.utils.Sequence):
    '''

    Data generator for Keras model
    Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Parameters
    ------------
    paths: list(str or path-like object)
        locations of each npz sample spaced with total duration
    targets: list(int)
        label encoded activity states
    mode: str
        CNN or RCNN
    td: int
        number of seconds per sample
    dt: float
        time to slice total_duration
    n_classes: int
        number of classes
    input_shape:  CNN: (time, feats, n_channels)
                 RCNN: (slice, time, feats, n_channels)
        slice : time feats sliced by delta_time
        time : step_size per second (10 ms)
        feats : freq_bins from stft or mfcc
        n_channels : 1
    epoch_frac: float
        percentage of data to train per epoch of total dataset
    batch_size: 32
        number of samples per batch
    suffle: True
        boolean to shuffle data between epochs

    '''
    def __init__(self, paths, targets, mode, td, dt, n_classes,
                 input_shape, epoch_frac=1.0, batch_size=32, shuffle=True):
        self.paths = paths
        self.targets = targets
        self.mode = mode
        self.td = td
        self.dt = dt
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.epoch_frac = epoch_frac
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.paths) / self.batch_size)*self.epoch_frac)


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        paths =   [self.paths[k] for k in indexes]
        targets = [self.targets[k] for k in indexes]

        X, y = self.__data_generation(paths, targets)

        return X, y


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, paths, targets):
        # Generates a batch of data by reading in npz until total duration is met
        if len(self.input_shape) == 3:
            X = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], 1), dtype=np.float32)

        elif len(self.input_shape) == 4:
            X = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1],
                          self.input_shape[2], 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, target) in enumerate(zip(paths, targets)):

            y = to_categorical(target, num_classes=self.n_classes)

            base = os.path.split(path)[0]
            start_ix = int(os.path.split(path)[-1].split('.npy')[0])
            frames = []
            for t in range(self.td):
                ix = str(start_ix + (t*100))
                path = os.path.join(base, ix+'.npy')
                x = np.load(path)
                frames.append(x)

            x = np.concatenate(frames, axis=0)

            if self.mode == 'CNN':
                X[i,] = np.expand_dims(x, axis=2)
                Y[i,] = y

            elif self.mode == 'RCNN':
                frames = []
                for z in range(0, self.input_shape[0]*self.input_shape[1], self.input_shape[1]):
                    _slice = x[z:z+self.input_shape[1],:]
                    _slice = np.expand_dims(_slice, axis=0)
                    frames.append(_slice)
                x = np.concatenate(frames, axis=0)
                x = np.expand_dims(x, axis=3)
                X[i,] = x
                Y[i,] = y

        return X, Y
