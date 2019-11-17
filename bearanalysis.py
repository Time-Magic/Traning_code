import numpy as np
from scipy.fftpack import rfft, fft, ifft, fftshift, fftfreq
from matplotlib import pyplot as plt
from scipy.io import loadmat
from os.path import dirname

samples = 120
data_bearing = np.zeros((samples * 5))
a = dirname(__file__)
f = a + '/CaseWesternReserveUniversityData/normal_2_99.mat'
bear_dict = loadmat(f)
data_bearing[0:samples] = np.transpose(bear_dict['X099_FE_time'][samples * 2:samples * 3])
fft_data_bearing = fft(data_bearing)
freq_fft_data_bearing = fftfreq(fft_data_bearing.size, 1 / 12000)
freq_limit = int(fft_data_bearing.size / 2)
signal_limit = freq_limit
plt.stem(freq_fft_data_bearing[:freq_limit], abs(fft_data_bearing[:freq_limit]), linefmt='', use_line_collection=True)
plt.show()
