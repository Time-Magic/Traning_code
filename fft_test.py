import numpy as np
from scipy.fftpack import fftfreq
from numpy.fft import fft
from matplotlib import pyplot as plt
from scipy.io import loadmat
from os.path import dirname

samples = 600
x = np.linspace(0, 1, samples)
data_bearing = np.zeros(samples * 3)
data_bearing[:samples] = 2 * np.sin(2 * np.pi * 200 * x) + 2 * np.sin(2 * np.pi * 100 * x)
fft_data_bearing = fft(data_bearing)
# real_fft_data_bearing=rfft(data_bearing)
freq_fft_data_bearing = fftfreq(fft_data_bearing.size, 1 / samples)
freq_limit = int(fft_data_bearing.size / 2) * 2
signal_limit = freq_limit
plt.stem(freq_fft_data_bearing[:freq_limit], abs(fft_data_bearing[:freq_limit]), linefmt='', use_line_collection=True)
# plt.stem(freq_fft_data_bearing[:freq_limit],abs(real_fft_data_bearing)[:freq_limit],use_line_collection=True)

plt.show()
