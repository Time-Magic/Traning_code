import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn import svm  # In ElasticNet,we have two important variable, alpha and l1_ratio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

bostondata = load_boston()
boston_X = bostondata.data
boston_y = bostondata.target
scale = StandardScaler()
scale.fit(boston_X)
boston_x = scale.transform(boston_X)
pca = PCA(n_components=3)
boston_x = pca.fit_transform(boston_x)
fig = plt.figure()
ax = plt.gca(projection='3d')
# ax.scatter(boston_x[:, 0], boston_x[:, 1], boston_x[:, 2], marker='o', c=boston_y)
x_tran, x_test, y_tran, y_test = train_test_split(boston_x, boston_y, test_size=0.3, random_state=42)
result = []
z = np.zeros(shape=(10, 10))
test_number = len(y_test)
for i in range(1, 11, 1):
    for j in range(1, 11, 1):
        clf = svm.SVR(C=i / 10, epsilon=j / 10,gamma='auto').fit(x_tran, y_tran)
        y_pre = clf.predict(x_test)
        result.append([i, j, clf.score(x_test,y_test)])
        z[i - 1, j - 1] = clf.score(x_test,y_test)
print(result)
x = np.linspace(1, 10, 10)
y = np.linspace(1, 10, 10)
X, Y = np.meshgrid(x / 10, y / 10)
ax.plot_surface(X, Y, z, cmap=cm.coolwarm)
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(5))
ax.set_xlabel('C')
ax.set_ylabel('eplison')
ax.set_zlabel('score')
plt.title('Boston_SVR')
plt.show()

# TODO:整理为函数或类形式。
#  由于SCR具有两个自由参数C及epsilon，未来需要在程序上进行两个参数方面的调参并给出相关结论
#  波士顿房价数据集是一定需要清洗的，预测问题太差了.
# 11123441
