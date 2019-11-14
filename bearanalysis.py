from cwru import CWRU
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt
from numpy import ndarray as np

bear_data = CWRU(exp='12FanEndFault', rpm='1730', length=500)
fft_bear_data = fft(bear_data.X_train)
plt.plot(fft_bear_data)
plt.show()
