import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from likeplotfunctions import plot3d
from dataprepare import boston_DBSCAN

x_boston, y_boston = boston_DBSCAN()
x_train, x_test, y_train, y_test = train_test_split(x_boston, y_boston, random_state=55, test_size=0.25)
clf = LinearRegression(n_jobs=4).fit(x_train, y_train)
print(clf.score(x_test, y_test))




'''
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
'''

# TODO:整理为函数或类形式。
#  由于SCR具有两个自由参数C及epsilon，未来需要在程序上进行两个参数方面的调参并给出相关结论
#  波士顿房价数据集是一定需要清洗的，预测问题太差了.
