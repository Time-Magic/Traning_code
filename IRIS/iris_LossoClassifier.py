import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import Lasso
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
ax = fig.add_subplot(111)
# ax.scatter(iris_x[:, 0], iris_x[:, 1], iris_x[:, 2], marker='o', c=iris_y)
x_tran, x_test, y_tran, y_test = train_test_split(iris_x, iris_y, test_size=0.3, random_state=42)
result = {}
test_number = len(y_test)
for i in range(1, 11, 1):
    clf = Lasso(alpha=i / 10).fit(x_tran, y_tran)
    y_pre = clf.predict(x_test)
    result[i / 10] = sum(m < 0.5 for m in abs(y_test - y_pre)) / test_number
print(result)
ax.plot(list(result.keys()), list(result.values()))
plt.show()

# TODO:整理为函数或类形式。Lasoo回归在此算例中呈现预测值形式
