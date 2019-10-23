import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import LogisticRegression
from numpy import ndarray
from mpl_toolkits.mplot3d import Axes3D

irisdata = load_iris()
iris_X = irisdata.data
iris_y = irisdata.target
scale = StandardScaler()
scale.fit(iris_X)
iris_x = scale.transform(iris_X)
pca = PCA(n_components=3)
iris_x = pca.fit_transform(iris_x)
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(iris_x[:, 0], iris_x[:, 1], iris_x[:, 2], marker='o', c=iris_y)
# plt.show()
x_tran, x_test, y_tran, y_test = train_test_split(iris_x, iris_y, test_size=0.3, random_state=42)
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(x_tran, y_tran)
y_pre = clf.predict(x_test)
yx = fig.add_subplot(122)
yx.scatter(y_test, y_pre, marker='o')
plt.show()
right = sum((y_pre - y_test) == 0)
right_pro = right / len(y_test)
print(right_pro)
