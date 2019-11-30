import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def spl_and_Std(X_iris, Y_iris):
    X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.33)
    scl = StandardScaler().fit(X_train)
    x_train = scl.transform(X_train)
    x_test = scl.transform(X_test)
    return x_train, x_test, y_train, y_test
def plot_func_region(x, flag, pre_func):
    x1_min, x1_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    x2_min, x2_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .01
    xx, xy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = pre_func(np.c_[xx.ravel(), xy.ravel()])
    z = Z.reshape(xx.shape)
    plt.contourf(xx, xy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=flag, cmap=plt.cm.Spectral)


def random_result(X_iris, Y_iris):
    result = np.zeros((100, 100))
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.33)
        scl = StandardScaler().fit(X_train)
        x_train = scl.transform(X_train)
        x_test = scl.transform(X_test)
        for j in range(100):
            clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')
            clf.fit(x_train, y_train)
            result[i][j] = (clf.score(x_test, y_test))
    print(result.mean(axis=1))
    return result


def random_result100(X_iris, Y_iris):
    result = np.zeros((100))
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.33)
        scl = StandardScaler().fit(X_train)
        x_train = scl.transform(X_train)
        x_test = scl.transform(X_test)
        clf = LogisticRegression(solver='sag', multi_class='multinomial')
        clf.fit(x_train, y_train)
        result[i] = (clf.score(x_test, y_test))
    print(result.mean(axis=1))
    return result


def sag_nopunish_random100(X_iris, Y_iris):
    result = np.zeros((100))
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.33)
        scl = StandardScaler().fit(X_train)
        x_train = scl.transform(X_train)
        x_test = scl.transform(X_test)
        clf = LogisticRegression(solver='saga', multi_class='multinomial', penalty='none', max_iter=10000)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def sag_elasticnet_random100(X_iris, Y_iris):
    result = np.zeros((100, 11))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std(X_iris, Y_iris)
        for j in range(11):
            clf = LogisticRegression(solver='saga', multi_class='multinomial', penalty='elasticnet', l1_ratio=j / 10,
                                     max_iter=10000)
            clf.fit(x_train, y_train)
            result[i][j] = clf.score(x_test, y_test)
    return result


def saga_elasticnet_strong_random100(X_iris, Y_iris):
    result = np.zeros((100, 11))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std(X_iris, Y_iris)
        for j in range(1, 12):
            clf = LogisticRegression(solver='saga', multi_class='multinomial', penalty='elasticnet', l1_ratio=0.7,
                                     max_iter=10000, C=j)
            clf.fit(x_train, y_train)
            result[i][j - 1] = clf.score(x_test, y_test)
    return result


def solver_elasticnet_strong_random100(X_iris, Y_iris):
    import time
    result = np.zeros((100, 5))
    time_consum = np.zeros((100, 5))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std(X_iris, Y_iris)
        clf1 = LogisticRegression(solver='liblinear', multi_class='ovr', penalty='l2', max_iter=10000)
        clf2 = LogisticRegression(solver='lbfgs', multi_class='ovr', penalty='l2', max_iter=10000)
        clf3 = LogisticRegression(solver='newton-cg', multi_class='ovr', penalty='l2', max_iter=10000)
        clf4 = LogisticRegression(solver='sag', multi_class='ovr', penalty='l2', max_iter=10000)
        clf5 = LogisticRegression(solver='saga', multi_class='ovr', penalty='l2', max_iter=10000)
        start = time.perf_counter()
        clf1.fit(x_train, y_train)
        end1 = time.perf_counter()
        time_consum[i][0] = end1 - start
        clf2.fit(x_train, y_train)
        end2 = time.perf_counter()
        time_consum[i][1] = end2 - end1
        clf3.fit(x_train, y_train)
        end3 = time.perf_counter()
        time_consum[i][2] = end3 - end2
        clf4.fit(x_train, y_train)
        end4 = time.perf_counter()
        time_consum[i][3] = end4 - end3
        clf5.fit(x_train, y_train)
        end5 = time.perf_counter()
        time_consum[i][4] = end5 - end4

        result[i][0] = clf1.score(x_test, y_test)
        result[i][1] = clf2.score(x_test, y_test)
        result[i][2] = clf3.score(x_test, y_test)
        result[i][3] = clf4.score(x_test, y_test)
        result[i][4] = clf5.score(x_test, y_test)
    return result, time_consum
