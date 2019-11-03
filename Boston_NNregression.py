import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler

bostondata = load_boston()  # 导入boston数据
boston_X = bostondata.data
boston_y = bostondata.target
scale_boston = StandardScaler()  # 标准化
scale_boston.fit(boston_X)
boston_x = scale_boston.transform(boston_X)
pca = PCA(n_components=2)  # 二维降维
pca.fit(boston_X)
dimesionpower = pca.explained_variance_ratio_
print(dimesionpower)
boston_x_train, boston_x_test, boston_y_train, boston_y_test = train_test_split(boston_x, boston_y, test_size=0.3,
                                                                                random_state=42)
result = {
    i: MLPRegressor(hidden_layer_sizes=(i * 3, i * 2, i * 1), max_iter=5000).fit(boston_x_train, boston_y_train).score(
        boston_x_test, boston_y_test) for i in
    range(1, 11, 1)}
print(result)
plt.plot(list(result.keys()), list(result.values()))
plt.show()
# 就结果来讲，神经网络随规模增大，效果先好后坏，在充分训练的情况下，神经网络的规模是要减小的。但是很奇怪，却并不很快下降
# todo:使用神经网络来看，随着
