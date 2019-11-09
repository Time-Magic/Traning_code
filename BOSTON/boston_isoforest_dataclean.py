import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.ensemble import IsolationForest

bostondata = load_boston()  # 导入boston数据
boston_X = bostondata.data
boston_y = bostondata.target
boston_fulldata = np.c_[boston_X, boston_y]
boston_scaler = IsolationForest(contamination='auto', behaviour='new', random_state=66)
boston_scaler.fit(boston_fulldata)
datacondition = boston_scaler.predict(boston_fulldata)
number = sum(datacondition == 1)
boston_X_clean = np.zeros((number, boston_X.shape[1]))
boston_y_clean = np.zeros((number, 1))
j = 0
for i in range(0, datacondition.shape[0], 1):
    if datacondition[i] == 1:
        boston_X_clean[j] = boston_X[i]
        boston_y_clean[j] = boston_y[i]
        j += 1
boston_X = boston_X_clean
boston_y = boston_y_clean

scale_boston = StandardScaler()  # 标准化
scale_boston.fit(boston_X)
boston_x = scale_boston.transform(boston_X)
pca = PCA(n_components=2)  # 二维降维
pca.fit(boston_X)
dimesionpower = pca.explained_variance_ratio_
print(dimesionpower)
boston_X = pca.transform(boston_X)  # 绘制二维价格特性
# pic_boston_data2D = plt.scatter(boston_X[:, 0], boston_X[:, 1], c=boston_y)
plt.show()
boston_x_train, boston_x_test, boston_y_train, boston_y_test = train_test_split(boston_x, boston_y, test_size=0.3,
                                                                                random_state=42)
boston_regression = LinearRegression().fit(boston_x_train, boston_y_train)  # 线性回归拟合并评分
print(boston_regression.score(boston_x_test, boston_y_test))
boston_y_pretict = boston_regression.predict(boston_x_test)
picture_boston_LinearRegression = plt.figure()
subpic_boston_LinearRegression = picture_boston_LinearRegression.add_subplot(111)
subpic_boston_LinearRegression.scatter(boston_y_test, boston_y_pretict)  # 绘制预测-实际价格参数
plt.show()
print(boston_regression.score(boston_x_test, boston_y_test))
# todo:整理为函数或类的形式。
# 使用独立森林算法进行异常值检测剔除，就结果来讲并没有使得预测精度得到显著提高，可能该问题其他方法比较合适
