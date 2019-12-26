from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
def spl_and_Std_boston(X_boston='Xboston', Y_boston='Yboston'):
    X_boston = np.loadtxt('Boston_data', delimiter=',')
    Y_boston = np.loadtxt('Boston_txt', delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(X_boston, Y_boston, test_size=0.33, random_state=42)
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
