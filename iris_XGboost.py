import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from xgboost import XGBClassifier
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
for i in range(1, 20, 1):
    clf = XGBClassifier(max_depth=3 * i, n_estimators=i * 10, n_jobs=4).fit(x_tran, y_tran)
    result[i * 10] = clf.score(x_test, y_test)
print(result)
ax.plot(list(result.keys()), list(result.values()))
plt.show()

# TODO:整理为函数或类形式。
# 就从结果来看,效果还是挺好的，但是没有想想中那么地好，正确率先增加后减小，总体正确率月为98%左右
