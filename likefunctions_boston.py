from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston


def spl_and_Std_boston(X_boston='Xboston', Y_boston='Yboston'):
    X_boston = load_boston().data
    Y_boston = load_boston().target
    X_train, X_test, y_train, y_test = train_test_split(X_boston, Y_boston, test_size=0.33)
    scl = StandardScaler().fit(X_train)
    x_train = scl.transform(X_train)
    x_test = scl.transform(X_test)
    return x_train, x_test, y_train, y_test
