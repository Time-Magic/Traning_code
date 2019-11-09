import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
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
    i: RandomForestRegressor(n_estimators=i * 10).fit(boston_x_train, boston_y_train).score(
        boston_x_test, boston_y_test) for i in
    range(1, 21, 1)}
print(result)
plt.plot(list(result.keys()), list(result.values()))
plt.show()
# todo:修改为函数或模型
# 随机森林的预测效果较好，至少是比传统的线性预测较好，具体提升可能需要进一步清洗数据集才可以实现。
