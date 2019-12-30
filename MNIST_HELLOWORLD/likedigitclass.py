from sklearn.datasets import load_digits

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, GridSearchCV
import numpy as np
from likefunctions_boston import plotlikeconfig
from pandas import DataFrame
from matplotlib import pyplot as plt

class DigitEnsembleClassifier():
    '''手写数据集集成分类器对象
    :parameter

    model:str
    指定处理模型

    split_ratio:float
    指定分割比例

    random_seed:int
    指定处理过程中的随机种子

    paramset:dic
    指定模型中规定的参数，将被传进分类器初始化中,采用字典型

    :returns

    Classifier:object
    一个集成了多种算法实现的手写数据集分类器
    '''

    def __init__(self, model, paramset='', cv=3):
        '''导入数据并格式化为可用的数据形式'''
        self.model = model
        self.x = load_digits()['data']
        self.y = load_digits()['target']
        self.param = paramset
        self.cv = cv
        self.normal()
        self.statu = 1
        if self.statu != 0:
            self.fit()

    def normal(self):
        '''数据预处理部分'''
        std = StandardScaler()
        std.fit_transform(self.x)

    def fit(self):
        '''该函数用于创建模型并进行调参拟合'''
        self.clf = GridSearchCV(self.model, self.param, cv=self.cv, return_train_score=True, error_score=0, n_jobs=-1)
        self.clf.fit(self.x, self.y)
        self.result = self.clf.cv_results_
        self.statu = 0

    def linedata(self, xlabel, const_param1='C', paramvalue1=1,
                 const_param2='solver', paramvalue2='saga', target='mean_test_score'):
        '''按照指定要求绘制线图图像
        :parameter

        xlabel:str
        可调自变量

        const_param1:str
        绘图需要固定的可调参数1

        paramvalue1:float/int/str
        可调参数1的固定值

         const_param2:str
        绘图需要固定的可调参数2

        paramvalue2:float/int/str
        可调参数2的固定值

        target:str
        目标参数值，将作为纵坐标数值

        :returns

        linex:float
        自变量参数值

        liney:float
        因变量参数值

        condition_label:str
        条件标签
        '''
        result = self.result
        const_param11 = 'param_' + const_param1
        const_param22 = 'param_' + const_param2
        param = 'param_' + xlabel
        idx1 = result[const_param11] == paramvalue1
        idx2 = result[const_param22] == paramvalue2
        idx = ~(~idx1 + ~idx2)
        linex = result[param][idx]
        liney = result[target][idx]
        condition_label = const_param1.capitalize() + '=' + paramvalue1 + '\n' + const_param2.capitalize() + '=' + paramvalue2 + '\n' + param.upper()
        return linex, liney, condition_label

    def export(self, name='data'):
        '''输出数据图表,传入参数无文件类型名
        :parameter

        name:str
        生成excel文件的名字
        '''
        result = DataFrame.from_dict(self.result)
        name += '.xls'
        result.to_excel(name)

# todo:需要添加的功能，数据集保存功能，把调整结果导出为文档，以备调参报告使用
