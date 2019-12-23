import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from likefunctions import plot_func_region

irisdata = load_iris()
iris_X = irisdata.data
iris_y = irisdata.target
scale = StandardScaler()
scale.fit(iris_X)
iris_x = scale.transform(iris_X)
pca = PCA(n_components=2)
iris_x = pca.fit_transform(iris_x)
x_tran, x_test, y_tran, y_test = train_test_split(iris_x, iris_y, test_size=0.3, random_state=42)
clf = svm.SVC(gamma='auto').fit(x_tran, y_tran)
y_pre = clf.predict(x_test)
plot_func_region(x_test, y_test, lambda x: clf.predict(x))
plt.title('SVC')
plt.show()
result = clf.score(x_test, y_test)
print(result)
# TODO:整理为函数或类形式.
#  新增加二维分类绘图函数plot_func_region()
