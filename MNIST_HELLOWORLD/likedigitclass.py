from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class DigitEnsembleClassifier():
    '''手写数据集集成分类器对象
    :parameter

    model:str
    指定处理模型

    split_ratio:float
    指定分割比例

    random_seed:int
    指定处理过程中的随机种子

    :returns

    Classifier:object
    一个集成了多种算法的手写数据集分类器
    '''

    def __init__(self, model='LogisticRegression', split_ratio=0.25, random_seed=42):
        '''导入数据并格式化为可用的数据形式'''
        self.model = model
        self.x = load_digits()['data']
        self.y = load_digits()['target']
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y)

    def normal(self):
        '''数据预处理部分'''
        pass

    def LR(self):
        '''逻辑回归处理'''
        self.clf = LogisticRegression(random_state=self.random_seed, solver='sag', multi_class='multinomial', n_jobs=4,
                                      max_iter=1000).fit(self.x_train, self.y_train)
        self.fitscore()

    def RC(self, alpha=0.1):
        '''岭回归处理'''
        self.clf = RidgeClassifier(alpha=alpha, random_state=self.random_seed).fit(self.x_train, self.y_train)
        self.fitscore()

    def XGBC(self):
        '''XGBoost分类器处理'''
        self.clf = XGBClassifier(max_depth=4, n_jobs=4, n_estimators=1000, random_state=self.random_seed).fit(
            self.x_train, self.y_train)
        self.fitscore()

    def RF(self):
        '''随机森林分类器处理'''
        self.clf = RandomForestClassifier(n_estimators=1000, max_depth=4, n_jobs=4, random_state=self.random_seed).fit(
            self.x_train, self.y_train)
        self.fitscore()

    def MLPC(self):
        '''多层感知机处理'''
        self.clf = MLPClassifier(hidden_layer_sizes=3, max_iter=10000, random_state=self.random_seed).fit(
            self.x_train, self.y_train)
        self.fitscore()

    def LC(self):
        '''Lasso分类器处理'''
        self.clf = Lasso(random_state=self.random_seed).fit(self.x_train, self.y_train)
        self.fitscore()

    def EN(self):
        '''弹性网络处理'''
        self.clf = ElasticNet(random_state=self.random_seed).fit(self.x_train, self.y_train)
        self.fitscore()

    def DT(self):
        '''单颗决策树处理'''
        self.clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=self.random_seed,
                                          max_leaf_nodes=200) \
            .fit(self.x_train, self.y_train)
        self.fitscore()

    def fitscore(self):
        '''给出模型拟合优度并输出'''
        self.goodness = self.clf.score(self.x_test, self.y_test)
        print(self.goodness)

        # todo:分析各个处理算法的可调参数，进行调参分析，以此基础绘制图像
