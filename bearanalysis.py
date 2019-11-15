import numpy as np
from scipy.fftpack import rfft, fft, ifft, fftshift, fftfreq
from matplotlib import pyplot as plt
from scipy.io import loadmat
from os.path import dirname

samples = 500
a = dirname(__file__)
f = a + '/CaseWesternReserveUniversityData/normal_2_99.mat'
bear_dict = loadmat(f)
data_bearing = bear_dict['X099_FE_time'][:samples]
x = np.linspace(0, 1, 1400)
# data_bearing=7*np.sin(2*np.pi*180*x)+2.8*np.sin(2*np.pi*390*x)

fft_data_bearing = fft(data_bearing)
freq_fft_data_bearing = fftfreq(fft_data_bearing.size, 1 / 12000)
freq_limit = int(samples / 2)
signal_limit = freq_limit
plt.plot(freq_fft_data_bearing[:freq_limit], abs(fft_data_bearing[:freq_limit]))
plt.show()
