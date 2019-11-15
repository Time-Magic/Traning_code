from cwru import CWRU
from numpy import ndarray as np
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from matplotlib import pyplot as plt
from scipy.io import loadmat
from os.path import dirname

a = dirname(__file__)
f = a + '/CaseWesternReserveUniversityData/normal_2_99.mat'
bear_dict = loadmat(f)
data_bearing = bear_dict['X099_FE_time']
# data_bearing=numpy.random.rand(samples)
fft_data_bearing = fft(data_bearing)
freq_fft_data_bearing = fftfreq(fft_data_bearing.size, 1 / 12000)
plt.stem(freq_fft_data_bearing, (fft_data_bearing.real ** 2 + fft_data_bearing.imag ** 2)
         , use_line_collection=True)
plt.show()
