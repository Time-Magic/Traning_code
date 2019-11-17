import numpy as np
import matplotlib.pyplot as plt

def plot_func_region(x, flag, pre_func):
    x1_min, x1_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    x2_min, x2_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .01
    xx, xy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = pre_func(np.c_[xx.ravel(), xy.ravel()])
    z = Z.reshape(xx.shape)
    plt.contourf(xx, xy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=flag, cmap=plt.cm.Spectral)
