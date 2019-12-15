import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def spl_and_Std(X_iris='Xiris', Y_iris='Y_iris'):
    X_iris = load_iris().data
    Y_iris = load_iris().target
    X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.33)
    scl = StandardScaler().fit(X_train)
    x_train = scl.transform(X_train)
    x_test = scl.transform(X_test)
    return x_train, x_test, y_train, y_test


# 读入iris数据集并进行划分,标准化
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
        clf1 = LogisticRegression(solver='liblinear', multi_class='ovr', penalty='l2', max_iter=10000,
                                  class_weight='balanced')
        clf2 = LogisticRegression(solver='lbfgs', multi_class='ovr', penalty='l2', max_iter=10000,
                                  class_weight='balanced')
        clf3 = LogisticRegression(solver='newton-cg', multi_class='ovr', penalty='l2', max_iter=10000,
                                  class_weight='balanced')
        clf4 = LogisticRegression(solver='sag', multi_class='ovr', penalty='l2', max_iter=10000,
                                  class_weight='balanced')
        clf5 = LogisticRegression(solver='saga', multi_class='ovr', penalty='l2', max_iter=10000,
                                  class_weight='balanced')
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


def tree_full_100_random():
    result = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced')
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def tree_depth_100_random(max_depth=2):
    result = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced', max_depth=max_depth)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def tree_leaf_100_random(min_leaf_samples=1):
    result = np.zeros(100)
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced',
                                     min_samples_leaf=min_leaf_samples)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def tree_impurity_100_random(impurity_decrease):
    result = np.zeros(100)
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced',
                                     min_impurity_decrease=impurity_decrease)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def randomforest_number_100_random(number=5):
    result = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = RandomForestClassifier(n_estimators=(number))
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def randomforest_feature_100_random(number=5):
    result = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = RandomForestClassifier(max_features=number, n_estimators=20)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def randomforest_depth_100_random(number=3):
    result = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = RandomForestClassifier(max_features=4, n_estimators=20, max_depth=number)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
    return result


def XGBoost_number_100_random(number=10):
    result = np.zeros((100))
    result_train = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = XGBClassifier(n_estimators=number)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
        result_train[i] = clf.score(x_train, y_train)
    return result, result_train


def XGBoost_gamma_100_random(gamma=0):
    result = np.zeros((100))
    result_train = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = XGBClassifier(n_estimators=40, gamma=gamma)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
        result_train[i] = clf.score(x_train, y_train)
    return result, result_train


def XGBoost_learning_100_random(learning_rate=0.1):
    result = np.zeros((100))
    result_train = np.zeros((100))
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = XGBClassifier(n_estimators=40, gamma=0.9, learning_rate=learning_rate)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
        result_train[i] = clf.score(x_train, y_train)
    return result, result_train


def XGBoost_learning_time_100_random(learning_rate=0.1):
    import time
    result = np.zeros((100))
    result_train = np.zeros((100))
    time_start = time.perf_counter()
    for i in range(100):
        x_train, x_test, y_train, y_test = spl_and_Std()
        clf = XGBClassifier(n_estimators=40, gamma=0.9, learning_rate=learning_rate)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
        result_train[i] = clf.score(x_train, y_train)
    time_end = time.perf_counter()
    times = time_start - time_end
    return result, result_train, times


def plot_area(x_train, x_test, y_test, predict):
    from sklearn.decomposition import PCA
    dcp = PCA(n_components=2)
    dcp.fit(x_train)
    x_test2 = dcp.transform(x_test)
    x1_min, x1_max = x_test2[:, 0].min() - .5, x_test2[:, 0].max() + .5
    x2_min, x2_max = x_test2[:, 1].min() - .5, x_test2[:, 1].max() + .5
    h = .01
    xx, xy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = predict(dcp.inverse_transform(np.c_[xx.ravel(), xy.ravel()]))
    z = Z.reshape(xx.shape)
    plt.contourf(xx, xy, z, cmap=plt.cm.Spectral)
    plt.scatter(x_test2[:, 0], x_test2[:, 1], c=y_test, cmap=plt.cm.Spectral)
