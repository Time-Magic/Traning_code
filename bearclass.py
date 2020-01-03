from MNIST_HELLOWORLD.likedigitclass import DigitEnsembleClassifier
from readdata import BearingDataReader
import numpy as np
from numpy.fft import fft
from scipy.signal import stft
from pywt import dwt, idwt


class BearingParam(DigitEnsembleClassifier):
    '''

    继承自BearingDataReader，读入轴承数据集，并进行一定预处理及参数交叉验证

    '''
    def __init__(self, model, dataset='fft', paramset='', cv=3):
        self.model = model
        origin_data = BearingDataReader()
        self.x = np.vstack([origin_data.X_train, origin_data.X_test])
        self.y = np.concatenate([origin_data.y_train, origin_data.y_test])
        self.param = paramset
        self.cv = cv
        self.prepare(kind=dataset)
        self.fit()

    def prepare(self, kind='fft'):
        if kind == 'fft':
            self.x = abs(fft(self.x)[:, :int(self.x.shape[1] / 2)])
        if kind == 'stft':
            f, t, z = stft(self.x)
            self.x = z.mean(axis=2)
        if kind == 'wavelet':
            coffes = dwt(self.x, wavelet='haar', axis=1)
            self.x = idwt(cA=coffes[0], wavelet='haar', axis=1)
            self.x = abs(fft(self.x))[:, :int(self.x.shape[1] / 2)]
