import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import ElasticNet  # In ElasticNet,we have two important variable, alpha and l1_ratio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

irisdata = load_iris()
iris_X = irisdata.data
iris_y = irisdata.target
scale = StandardScaler()
scale.fit(iris_X)
iris_x = scale.transform(iris_X)
pca = PCA(n_components=3)
iris_x = pca.fit_transform(iris_x)
fig = plt.figure()
ax = plt.gca(projection='3d')
# ax.scatter(iris_x[:, 0], iris_x[:, 1], iris_x[:, 2], marker='o', c=iris_y)
x_tran, x_test, y_tran, y_test = train_test_split(iris_x, iris_y, test_size=0.3, random_state=42)
result = []
z = np.zeros(shape=(10, 10))
test_number = len(y_test)
for i in range(1, 11, 1):
    for j in range(1, 11, 1):
        clf = ElasticNet(alpha=i / 10, l1_ratio=j / 10).fit(x_tran, y_tran)
        y_pre = clf.predict(x_test)
        result.append([i, j, (sum(m < 0.5 for m in abs(y_test - y_pre)) / test_number)])
        z[i - 1, j - 1] = (sum(m < 0.5 for m in abs(y_test - y_pre)) / test_number)
print(result)
x = np.linspace(1, 10, 10)
y = np.linspace(1, 10, 10)
X, Y = np.meshgrid(x / 10, y / 10)
ax.plot_surface(X, Y, z, cmap=cm.coolwarm)
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(5))

plt.show()

# TODO:整理为函数或类形式。
#  ElasticNet回归在此算例中呈现预测值形式
#  由于Elastic具有两个参数，未来需要在程序上进行两个参数方面的调参并给出相关结论。
