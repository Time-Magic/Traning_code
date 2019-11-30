import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from likefunctions import solver_elasticnet_strong_random100

datas = load_iris()
X_iris = datas.data
Y_iris = datas.target
xlabel_solver = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
result_solver, time = solver_elasticnet_strong_random100(X_iris, Y_iris)
result_solver100 = result_solver.mean(axis=0)
plt.bar(xlabel_solver, time.mean(axis=0))
plt.xlabel('solver')
plt.ylabel('time consuming')
plt.show()
