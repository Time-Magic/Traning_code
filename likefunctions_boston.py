from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from math import ceil


def spl_and_Std_boston(X_boston='Xboston', Y_boston='Yboston', origin=False, factual=0.33):
    X_boston = np.loadtxt('Boston_data', delimiter=',')
    Y_boston = np.loadtxt('Boston_txt', delimiter=',')
    if origin == True:
        X_boston = load_boston().data
        Y_boston = load_boston().target
    X_train, X_test, y_train, y_test = train_test_split(X_boston, Y_boston, test_size=1 - factual, random_state=42)
    scl = StandardScaler().fit(X_train)
    x_train = scl.transform(X_train)
    x_test = scl.transform(X_test)
    return x_train, x_test, y_train, y_test


def plotlikeconfig(xlabel='Parameter', ylabel='Accuracy', title='Parameter Adjust', font='Times New Roman', dpi=200):
    plt.rc('font', family='Times New Roman')
    plt.figure(dpi=dpi, linewidth=10)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=18)
    plt.tick_params(direction='in', width=2, top=True, right=True, labelsize=12)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


def Dtr_depth_random_100(Depth=3):
    result = np.zeros((1))
    result_train = np.zeros((1))
    for i in range(1):
        x_train, x_test, y_train, y_test = spl_and_Std_boston()
        clf = DecisionTreeRegressor(max_depth=Depth)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
        result_train[i] = clf.score(x_train, y_train)
    return result, result_train


def spl_and_Std_pca_boston(dimension=5):
    X_boston = np.loadtxt('Boston_data', delimiter=',')
    Y_boston = np.loadtxt('Boston_txt', delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(X_boston, Y_boston, test_size=0.33, random_state=42)
    scl = StandardScaler().fit(X_train)
    x_train = scl.transform(X_train)
    x_test = scl.transform(X_test)
    pca = PCA(n_components=dimension)
    pca.fit(x_train)
    pca.transform(x_train)
    pca.transform(x_test)
    return x_train, x_test, y_train, y_test


def Dtr_depth_pca_random_100(Depth=5):
    result = np.zeros((1))
    result_train = np.zeros((1))
    for i in range(1):
        x_train, x_test, y_train, y_test = spl_and_Std_pca_boston()
        clf = DecisionTreeRegressor(max_depth=Depth)
        clf.fit(x_train, y_train)
        result[i] = clf.score(x_test, y_test)
        result_train[i] = clf.score(x_train, y_train)
    return result, result_train


def Dtr_pca(dimension=5):
    x_train, x_test, y_train, y_test = spl_and_Std_pca_boston(dimension=dimension)
    clf = DecisionTreeRegressor(max_depth=5)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def Dtr_lossf(function='mse'):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = DecisionTreeRegressor(max_depth=5)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def RandomForestRegression_number(number=100):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = RandomForestRegressor(n_estimators=number)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def xgboost_number(number=100):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = XGBRegressor(n_estimators=number, objective='reg:squarederror')
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def xgboost_depth(depth=100):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = XGBRegressor(n_estimators=100, max_depth=depth, objective='reg:squarederror')
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def svr_epsilon(epsilon=0.1):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = SVR(epsilon=epsilon)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def svr_C(C=1):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = SVR(epsilon=0.7, C=C)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def svr_kernel(kernel='rbf'):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    clf = SVR(epsilon=0.7, C=15, kernel=kernel)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def ann_depth(depth=3, init=5):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    layer = np.array([ceil(init * 1.2 ** i) for i in range(1, depth + 1)])
    pca = PCA(n_components=2)
    pca.fit_transform(x_train)
    pca.transform(x_test)
    clf = MLPRegressor(hidden_layer_sizes=layer, max_iter=100000, solver='lbfgs')
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train


def ann_L2(depth=3, init=5, L2=0.001):
    x_train, x_test, y_train, y_test = spl_and_Std_boston()
    layer = np.array([ceil(init * 0.9 ** i) for i in range(1, depth + 1)])
    pca = PCA(n_components=13)
    pca.fit_transform(x_train)
    pca.transform(x_test)
    clf = MLPRegressor(hidden_layer_sizes=layer, max_iter=100000, solver='lbfgs', alpha=L2)
    clf.fit(x_train, y_train)
    result = np.array(clf.score(x_test, y_test))
    result_train = np.array(clf.score(x_train, y_train))
    return result, result_train
