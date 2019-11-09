from sklearn.cluster import DBSCAN
from sklearn.datasets import load_boston
from sklearn.preprocessing.data import StandardScaler
import numpy as np
from sklearn.decomposition import PCA


def boston_DBSCAN(class_num=0):
    '''给出Boston房价数据集中class_num类，数据形式为归一化数据列
    :parameter
    ————
    class_num:类号，读取函数所返回的类别号
    :returns
    ————
    x_boston:波士顿数据集中class_num类的自变量，归一化数据列，共13列
    y_boston:波士顿数据集中class_num类自变量，归一化数据列，共1列
    '''
    # 读取全数据集
    bostondata = load_boston()
    boston_X = bostondata.data
    boston_y = bostondata.target
    boston_full = np.c_[boston_X, boston_y]
    # 进行全数据集归一化
    scale = StandardScaler()
    boston_full = scale.fit_transform(boston_full)
    # 数据集降维为3维，方便可视化调参。
    pca = PCA(n_components=3)
    boston_full3 = pca.fit_transform(boston_full)
    # 分类
    clt = DBSCAN(eps=0.8, min_samples=5, n_jobs=4)
    label3 = clt.fit_predict(X=boston_full3)
    # 给定输出数据
    group0_boston = boston_full[label3 == 0]
    x_boston = group0_boston[:, 0:-2]
    y_boston = group0_boston[:, -1]
    return x_boston, y_boston
