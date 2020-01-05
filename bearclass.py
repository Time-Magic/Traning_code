from MNIST_HELLOWORLD.likedigitclass import DigitEnsembleClassifier
from readdata import BearingDataReader
import numpy as np
from numpy.fft import fft
from scipy.signal import stft
from pywt import dwt, idwt, wavedec, dwt2


class BearingParam(DigitEnsembleClassifier):
    '''

    继承自BearingDataReader，读入轴承数据集，并进行一定预处理及参数交叉验证

    '''
    def __init__(self, model, dataset='fft', paramset='', cv=3):
        self.model = model
        origin_data = BearingDataReader(ratio=0.5)
        self.x = origin_data.X_train
        self.y = origin_data.y_train
        self.param = paramset
        self.cv = cv
        self.prepare(kind=dataset)
        self.fit()

    def prepare(self, kind='fft'):
        temp = self.x
        if kind == 'fft':
            fft_temp = abs(fft(temp)[:, :int(temp.shape[1] / 2)])
            dec_temp = wavedec(fft_temp, wavelet='haar', level=2, axis=1)
            self.x = dec_temp[0]
        if kind == 'stft':
            f, t, z = stft(temp, fs=12000, nperseg=256)
            self.x = abs(z).mean(axis=2)
            dec_temp = wavedec(fft_temp, wavelet='haar', level=1, axis=1)
            self.x = dec_temp[0]
        if kind == 'wavelet':
            coffes = dwt(temp, wavelet='haar', axis=1)
            idwt_temp = idwt(cA=coffes[0], cD=None, wavelet='haar', axis=1)
            ifft_temp = abs(fft(idwt_temp))[:, :int(temp.shape[1] / 2)]
            dec_temp = wavedec(ifft_temp, wavelet='haar', level=2, axis=1)
            self.x = dec_temp[0]
