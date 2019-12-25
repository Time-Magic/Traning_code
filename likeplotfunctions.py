import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3d(x, y):
    '''用于绘制三维散点图，默认采用coolwarm染色方案
    参数
    ——
    x:位置参数，形如n_sample*3,为绘制点的坐标
    y:值变量，形如n_sample*1,为绘制点的颜色值，映射至蓝红两色之间
    '''
    fig = plt.figure()
    # ax = plt.gca(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o', c=y, cmap=cm.coolwarm)


d
