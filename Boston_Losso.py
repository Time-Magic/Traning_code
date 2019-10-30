import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
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
result = {i / 10: Lasso(alpha=i / 10).fit(boston_x_train, boston_y_train).score(boston_x_test, boston_y_test) for i in
          range(1, 11, 1)}
plt.plot(list(result.keys()), list(result.values()))
plt.show()
print(result)  # 就结果来讲，正则修正系数越小越好，当然这也是在剔除异常值之前的了，以alpha=0为佳
# todo:使用聚类法整理原始数据，剔除一些异常数据信息
