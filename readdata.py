import os
import random
import numpy as np
from scipy.io import loadmat


class BearingDataReader:
    def __init__(self, frequency='12k', location='Drive', length=1200, ratio=0.75):
        self.frequency = frequency
        self.location = location
        self.length = length
        self.ratio = ratio
        rdir = (os.path.dirname(__file__) + '/CaseWesternReserveUniversityData/Data')
        files = os.listdir(rdir)
        lines = [x for x in files if frequency + '_' + location in x]
        self._load_and_slice_data(rdir, lines)
        self._shuffle()
        self.label = [x.split('_')[-1].split('.')[0] for x in lines]

    def _load_and_slice_data(self, rdir, lines):
        self.X_train = np.zeros((0, self.length))
        self.X_test = np.zeros((0, self.length))
        self.y_train = []
        self.y_test = []
        for idx, info in enumerate(lines):
            fpath = os.path.join(rdir, info)
            mat_dict = loadmat(fpath)
            key = [x for x in mat_dict.keys() if 'DE_time' in x]
            time_series = mat_dict[key[0]][:, 0]
            idx_last = -(time_series.shape[0] % self.length)
            clips = time_series[:idx_last].reshape(-1, self.length)
            n = clips.shape[0]
            n_split = int(n * self.ratio)
            self.X_train = np.vstack((self.X_train, clips[:n_split]))
            self.X_test = np.vstack((self.X_test, clips[n_split:]))
            self.y_train += [idx] * n_split
            self.y_test += [idx] * (clips.shape[0] - n_split)

    def _shuffle(self):
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = tuple(self.y_train[i] for i in index)
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = tuple(self.y_test[i] for i in index)
