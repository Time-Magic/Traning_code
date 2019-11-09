import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
for i in range(1, 50, 1):
    clf = RandomForestClassifier(n_estimators=i * 10).fit(x_tran, y_tran)
    result[i * 10] = clf.score(x_test, y_test)
print(result)
ax.plot(list(result.keys()), list(result.values()))
plt.show()

# TODO:整理为函数或类形式。
# 就从结果来看，随着分类器的增加，预测效果先好后坏，在经过大的降幅过后又重新变好，虽然总体来讲总是能保持不错的效果，但是
# 波动较为剧烈。
