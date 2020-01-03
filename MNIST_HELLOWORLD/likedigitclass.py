from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
from pandas import read_excel
class DigitEnsembleClassifier():
    '''手写数据集集成分类器对象,传入模型及参数进行网格调参并将网格结果保存为文件，默认保存地址为项目地址
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

    def export(self, name='data'):
        '''输出数据图表,传入参数无文件类型名
        :parameter

        name:str
        生成excel文件的名字
        '''
        result = DataFrame.from_dict(self.result)
        name += '.xls'
        result.to_excel(name)



def linedata(xlabel, name='test', const_param1='C', paramvalue1=1,
             const_param2='solver', paramvalue2='saga', target='mean_test_score'):
    '''按照指定要求绘制线图图像
    :parameter

    name:str
    数据文件名

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
    result = read_excel(name + '.xls')
    const_param11 = 'param_' + const_param1
    const_param22 = 'param_' + const_param2
    param = 'param_' + xlabel
    idx1 = result[const_param11] == paramvalue1
    idx2 = result[const_param22] == paramvalue2
    idx = ~(~idx1 | ~idx2)
    linex = result[param][idx]
    liney = result[target][idx]
    condition_label = const_param1.capitalize() + '=' + str(paramvalue1) + '\n' + const_param2.capitalize() + '=' + str(
        paramvalue2) + '\n' + target.upper()
    return linex, liney, condition_label
# todo:类中linedata方法删除，单独作为linedata函数进行使用
