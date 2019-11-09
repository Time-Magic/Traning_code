import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from BOSTON.dataprepare import boston_DBSCAN

boston_x, boston_y = boston_DBSCAN()
pca = PCA(n_components=2)  # 二维降维
pca.fit_transform(boston_x)
dimesionpower = pca.explained_variance_ratio_
print(dimesionpower)
boston_x_train, boston_x_test, boston_y_train, boston_y_test = train_test_split(boston_x, boston_y, test_size=0.25,
                                                                                random_state=9)
result = {
    i: MLPRegressor(hidden_layer_sizes=(i * 3, i * 2, i * 1), max_iter=5000).fit(boston_x_train, boston_y_train).score(
        boston_x_test, boston_y_test) for i in
    range(1, 11, 1)}
print(result)
plt.plot(list(result.keys()), list(result.values()))
plt.show()
# todo:使用神经网络来看，随着模增大，效果先好后坏，在充分训练的情况下，神经网络的规模是要减小的。但是很奇怪，却并不很快
#  下降同时，神经网络的效果是要相对好于传统模型的。
