import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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
boston_X = pca.transform(boston_X)  # 绘制二维价格特性
pic_boston_data2D = plt.scatter(boston_X[:, 0], boston_X[:, 1], c=boston_y)
plt.show()
boston_x_train, boston_x_test, boston_y_train, boston_y_test = train_test_split(boston_x, boston_y, test_size=0.3,
                                                                                random_state=42)
boston_regression = LinearRegression().fit(boston_x_train, boston_y_train)  # 线性回归拟合并评分
print(boston_regression.score(boston_x_test, boston_y_test))
boston_y_pretict = boston_regression.predict(boston_x_test)
picture_boston_LinearRegression = plt.figure()
subpic_boston_LinearRegression = picture_boston_LinearRegression.add_subplot(111)
subpic_boston_LinearRegression.scatter(boston_y_test, boston_y_pretict)  # 绘制预测实际价格参数
plt.show()
# todo:使用聚类法整理原始数据，剔除一些异常数据信息