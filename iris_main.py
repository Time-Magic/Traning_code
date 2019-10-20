from sklearn.datasets import load_iris
from sklearn.preprocessing.data import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

irisdata = load_iris()
iris_X = irisdata.data
iris_y = irisdata.target
scale = StandardScaler()
scale.fit(iris_X)
iris_x = scale.transform(iris_X)
pca = PCA(n_components=3)
