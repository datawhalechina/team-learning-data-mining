
# Task2 数据分析

此部分为零基础入门金融风控的 Task2 数据分析部分，带你来了解数据，熟悉数据，为后续的特征工程做准备，欢迎大家后续多多交流。

赛题：零基础入门数据挖掘 - 零基础入门金融风控之贷款违约

目的：
- 1.EDA价值主要在于熟悉了解整个数据集的基本情况（缺失值，异常值），对数据集进行验证是否可以进行接下来的机器学习或者深度学习建模.

- 2.了解变量间的相互关系、变量与预测值之间的存在关系。

- 3.为特征工程做准备

预测： https://tianchi.aliyun.com/competition/entrance/531830/introduction

## 2.1 学习目标

- 学习如何对数据集整体概况进行分析，包括数据集的基本情况（缺失值，异常值）
- 学习了解变量间的相互关系、变量与预测值之间的存在关系
- 完成相应学习打卡任务 
 

## 2.2 内容介绍

- 数据总体了解：
  - 读取数据集并了解数据集大小，原始特征维度；
  - 通过info熟悉数据类型；
  - 粗略查看数据集中各特征基本统计量；
- 缺失值和唯一值：
  - 查看数据缺失值情况
  - 查看唯一值特征情况
- 深入数据-查看数据类型
  - 类别型数据
  - 数值型数据
    - 离散数值型数据
    - 连续数值型数据
- 数据间相关关系
  - 特征和特征之间关系
  - 特征和目标变量之间关系
- 用pandas_profiling生成数据报告


## 2.3 代码示例

###  2.3.1 导入数据分析及可视化过程需要的库


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
```

    /Users/exudingtao/opt/anaconda3/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    

以上库都是pip install 安装就好，如果本机有python2,python3两个python环境傻傻分不清哪个的话,可以pip3 install 。或者直接在notebook中'!pip3 install ****'安装。

说明：

本次数据分析探索，尤其可视化部分均选取某些特定变量进行了举例，所以它只是一个方法的展示而不是整个赛题数据分析的解决方案。

###  2.3.2 读取文件


```python
data_train = pd.read_csv('./train.csv')
```


```python
data_test_a = pd.read_csv('./testA.csv')
```

读取文件的拓展知识：

- pandas读取数据时相对路径载入报错时，尝试使用os.getcwd()查看当前工作目录。
- TSV与CSV的区别：
    - 从名称上即可知道，TSV是用制表符（Tab,'\t'）作为字段值的分隔符；CSV是用半角逗号（','）作为字段值的分隔符；
    - Python对TSV文件的支持：
      Python的csv模块准确的讲应该叫做dsv模块，因为它实际上是支持范式的分隔符分隔值文件（DSV，delimiter-separated values）的。
      delimiter参数值默认为半角逗号，即默认将被处理文件视为CSV。当delimiter='\t'时，被处理文件就是TSV。
- 读取文件的部分（适用于文件特别大的场景）
    - 通过nrows参数，来设置读取文件的前多少行，nrows是一个大于等于0的整数。
    - 分块读取 


```python
data_train_sample = pd.read_csv("./train.csv",nrows=5)
```


```python
#设置chunksize参数，来控制每次迭代数据的大小
chunker = pd.read_csv("./train.csv",chunksize=5)
for item in chunker:
    print(type(item))
    #<class 'pandas.core.frame.DataFrame'>
    print(len(item))
    #5
```

### 2.3.3 总体了解


查看数据集的样本个数和原始特征维度


```python
data_test_a.shape
```




    (200000, 48)




```python
data_train.shape
```




    (800000, 47)




```python
data_train.columns
```




    Index(['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'grade',
           'subGrade', 'employmentTitle', 'employmentLength', 'homeOwnership',
           'annualIncome', 'verificationStatus', 'issueDate', 'isDefault',
           'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years',
           'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec',
           'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
           'initialListStatus', 'applicationType', 'earliesCreditLine', 'title',
           'policyCode', 'n0', 'n1', 'n2', 'n2.1', 'n4', 'n5', 'n6', 'n7', 'n8',
           'n9', 'n10', 'n11', 'n12', 'n13', 'n14'],
          dtype='object')



查看一下具体的列名，赛题理解部分已经给出具体的特征含义，这里方便阅读再给一下：
- id	为贷款清单分配的唯一信用证标识
- loanAmnt	贷款金额
- term	贷款期限（year）
- interestRate	贷款利率
- installment	分期付款金额
- grade	贷款等级
- subGrade	贷款等级之子级
- employmentTitle	就业职称
- employmentLength	就业年限（年）
- homeOwnership	借款人在登记时提供的房屋所有权状况
- annualIncome	年收入
- verificationStatus	验证状态
- issueDate	贷款发放的月份
- purpose	借款人在贷款申请时的贷款用途类别
- postCode	借款人在贷款申请中提供的邮政编码的前3位数字
- regionCode	地区编码
- dti	债务收入比
- delinquency_2years	借款人过去2年信用档案中逾期30天以上的违约事件数
- ficoRangeLow	借款人在贷款发放时的fico所属的下限范围
- ficoRangeHigh	借款人在贷款发放时的fico所属的上限范围
- openAcc	借款人信用档案中未结信用额度的数量
- pubRec	贬损公共记录的数量
- pubRecBankruptcies	公开记录清除的数量
- revolBal	信贷周转余额合计
- revolUtil	循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额
- totalAcc	借款人信用档案中当前的信用额度总数
- initialListStatus	贷款的初始列表状态
- applicationType	表明贷款是个人申请还是与两个共同借款人的联合申请
- earliesCreditLine	借款人最早报告的信用额度开立的月份
- title	借款人提供的贷款名称
- policyCode	公开可用的策略_代码=1新产品不公开可用的策略_代码=2
- n系列匿名特征	匿名特征n0-n14，为一些贷款人行为计数特征的处理

通过info()来熟悉数据类型


```python
data_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 800000 entries, 0 to 799999
    Data columns (total 47 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   id                  800000 non-null  int64  
     1   loanAmnt            800000 non-null  float64
     2   term                800000 non-null  int64  
     3   interestRate        800000 non-null  float64
     4   installment         800000 non-null  float64
     5   grade               800000 non-null  object 
     6   subGrade            800000 non-null  object 
     7   employmentTitle     799999 non-null  float64
     8   employmentLength    753201 non-null  object 
     9   homeOwnership       800000 non-null  int64  
     10  annualIncome        800000 non-null  float64
     11  verificationStatus  800000 non-null  int64  
     12  issueDate           800000 non-null  object 
     13  isDefault           800000 non-null  int64  
     14  purpose             800000 non-null  int64  
     15  postCode            799999 non-null  float64
     16  regionCode          800000 non-null  int64  
     17  dti                 799761 non-null  float64
     18  delinquency_2years  800000 non-null  float64
     19  ficoRangeLow        800000 non-null  float64
     20  ficoRangeHigh       800000 non-null  float64
     21  openAcc             800000 non-null  float64
     22  pubRec              800000 non-null  float64
     23  pubRecBankruptcies  799595 non-null  float64
     24  revolBal            800000 non-null  float64
     25  revolUtil           799469 non-null  float64
     26  totalAcc            800000 non-null  float64
     27  initialListStatus   800000 non-null  int64  
     28  applicationType     800000 non-null  int64  
     29  earliesCreditLine   800000 non-null  object 
     30  title               799999 non-null  float64
     31  policyCode          800000 non-null  float64
     32  n0                  759730 non-null  float64
     33  n1                  759730 non-null  float64
     34  n2                  759730 non-null  float64
     35  n2.1                759730 non-null  float64
     36  n4                  766761 non-null  float64
     37  n5                  759730 non-null  float64
     38  n6                  759730 non-null  float64
     39  n7                  759730 non-null  float64
     40  n8                  759729 non-null  float64
     41  n9                  759730 non-null  float64
     42  n10                 766761 non-null  float64
     43  n11                 730248 non-null  float64
     44  n12                 759730 non-null  float64
     45  n13                 759730 non-null  float64
     46  n14                 759730 non-null  float64
    dtypes: float64(33), int64(9), object(5)
    memory usage: 286.9+ MB
    

总体粗略的查看数据集各个特征的一些基本统计量


```python
data_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loanAmnt</th>
      <th>term</th>
      <th>interestRate</th>
      <th>installment</th>
      <th>employmentTitle</th>
      <th>homeOwnership</th>
      <th>annualIncome</th>
      <th>verificationStatus</th>
      <th>isDefault</th>
      <th>...</th>
      <th>n5</th>
      <th>n6</th>
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>799999.000000</td>
      <td>800000.000000</td>
      <td>8.000000e+05</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>...</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
      <td>759729.000000</td>
      <td>759730.000000</td>
      <td>766761.000000</td>
      <td>730248.000000</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>399999.500000</td>
      <td>14416.818875</td>
      <td>3.482745</td>
      <td>13.238391</td>
      <td>437.947723</td>
      <td>72005.351714</td>
      <td>0.614213</td>
      <td>7.613391e+04</td>
      <td>1.009683</td>
      <td>0.199513</td>
      <td>...</td>
      <td>8.107937</td>
      <td>8.575994</td>
      <td>8.282953</td>
      <td>14.622488</td>
      <td>5.592345</td>
      <td>11.643896</td>
      <td>0.000815</td>
      <td>0.003384</td>
      <td>0.089366</td>
      <td>2.178606</td>
    </tr>
    <tr>
      <th>std</th>
      <td>230940.252015</td>
      <td>8716.086178</td>
      <td>0.855832</td>
      <td>4.765757</td>
      <td>261.460393</td>
      <td>106585.640204</td>
      <td>0.675749</td>
      <td>6.894751e+04</td>
      <td>0.782716</td>
      <td>0.399634</td>
      <td>...</td>
      <td>4.799210</td>
      <td>7.400536</td>
      <td>4.561689</td>
      <td>8.124610</td>
      <td>3.216184</td>
      <td>5.484104</td>
      <td>0.030075</td>
      <td>0.062041</td>
      <td>0.509069</td>
      <td>1.844377</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>500.000000</td>
      <td>3.000000</td>
      <td>5.310000</td>
      <td>15.690000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>199999.750000</td>
      <td>8000.000000</td>
      <td>3.000000</td>
      <td>9.750000</td>
      <td>248.450000</td>
      <td>427.000000</td>
      <td>0.000000</td>
      <td>4.560000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>399999.500000</td>
      <td>12000.000000</td>
      <td>3.000000</td>
      <td>12.740000</td>
      <td>375.135000</td>
      <td>7755.000000</td>
      <td>1.000000</td>
      <td>6.500000e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>599999.250000</td>
      <td>20000.000000</td>
      <td>3.000000</td>
      <td>15.990000</td>
      <td>580.710000</td>
      <td>117663.500000</td>
      <td>1.000000</td>
      <td>9.000000e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>10.000000</td>
      <td>19.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>799999.000000</td>
      <td>40000.000000</td>
      <td>5.000000</td>
      <td>30.990000</td>
      <td>1715.420000</td>
      <td>378351.000000</td>
      <td>5.000000</td>
      <td>1.099920e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>70.000000</td>
      <td>132.000000</td>
      <td>79.000000</td>
      <td>128.000000</td>
      <td>45.000000</td>
      <td>82.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>39.000000</td>
      <td>30.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 42 columns</p>
</div>




```python
data_train.head(3).append(data_train.tail(3))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loanAmnt</th>
      <th>term</th>
      <th>interestRate</th>
      <th>installment</th>
      <th>grade</th>
      <th>subGrade</th>
      <th>employmentTitle</th>
      <th>employmentLength</th>
      <th>homeOwnership</th>
      <th>...</th>
      <th>n5</th>
      <th>n6</th>
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>35000.0</td>
      <td>5</td>
      <td>19.52</td>
      <td>917.97</td>
      <td>E</td>
      <td>E2</td>
      <td>320.0</td>
      <td>2 years</td>
      <td>2</td>
      <td>...</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>18000.0</td>
      <td>5</td>
      <td>18.49</td>
      <td>461.90</td>
      <td>D</td>
      <td>D2</td>
      <td>219843.0</td>
      <td>5 years</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>12000.0</td>
      <td>5</td>
      <td>16.99</td>
      <td>298.17</td>
      <td>D</td>
      <td>D3</td>
      <td>31698.0</td>
      <td>8 years</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>799997</th>
      <td>799997</td>
      <td>6000.0</td>
      <td>3</td>
      <td>13.33</td>
      <td>203.12</td>
      <td>C</td>
      <td>C3</td>
      <td>2582.0</td>
      <td>10+ years</td>
      <td>1</td>
      <td>...</td>
      <td>4.0</td>
      <td>26.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>799998</th>
      <td>799998</td>
      <td>19200.0</td>
      <td>3</td>
      <td>6.92</td>
      <td>592.14</td>
      <td>A</td>
      <td>A4</td>
      <td>151.0</td>
      <td>10+ years</td>
      <td>0</td>
      <td>...</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>799999</th>
      <td>799999</td>
      <td>9000.0</td>
      <td>3</td>
      <td>11.06</td>
      <td>294.91</td>
      <td>B</td>
      <td>B3</td>
      <td>13.0</td>
      <td>5 years</td>
      <td>0</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 47 columns</p>
</div>



### 2.3.4 查看数据集中特征缺失值，唯一值等

查看缺失值


```python
print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')
```

    There are 22 columns in train dataset with missing values.
    

上面得到训练集有22列特征有缺失值，进一步查看缺失特征中缺失率大于50%的特征


```python
have_null_fea_dict = (data_train.isnull().sum()/len(data_train)).to_dict()
fea_null_moreThanHalf = {}
for key,value in have_null_fea_dict.items():
    if value > 0.5:
        fea_null_moreThanHalf[key] = value
```


```python
fea_null_moreThanHalf
```




    {}



具体的查看缺失特征及缺失率


```python
# nan可视化
missing = data_train.isnull().sum()/len(data_train)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1229ab890>




![output_34_1.png](https://img-blog.csdnimg.cn/20200905092323179.png)


了解哪些列存在 “nan”, 并可以把nan的个数打印，主要的目的在于 nan存在的个数是否真的很大，如果很小一般选择填充，如果使用lgb等树模型可以直接空缺，让树自己去优化，但如果nan存在的过多、可以考虑删掉

查看训练集测试集中特征属性只有一值的特征


```python
one_value_fea = [col for col in data_train.columns if data_train[col].nunique() <= 1]
```


```python
one_value_fea_test = [col for col in data_test_a.columns if data_test_a[col].nunique() <= 1]
```


```python
one_value_fea
```




    ['policyCode']




```python
one_value_fea_test
```




    ['policyCode']




```python
print(f'There are {len(one_value_fea)} columns in train dataset with one unique value.')
print(f'There are {len(one_value_fea_test)} columns in test dataset with one unique value.')
```

    There are 1 columns in train dataset with one unique value.
    There are 1 columns in test dataset with one unique value.
    

总结：

47列数据中有22列都缺少数据，这在现实世界中很正常。‘policyCode’具有一个唯一值（或全部缺失）。有很多连续变量和一些分类变量。

### 2.3.5 查看特征的数值类型有哪些，对象类型有哪些
- 特征一般都是由类别型特征和数值型特征组成
- 类别型特征有时具有非数值关系，有时也具有数值关系。比如‘grade’中的等级A，B，C等，是否只是单纯的分类，还是A优于其他要结合业务判断。
- 数值型特征本是可以直接入模的，但往往风控人员要对其做分箱，转化为WOE编码进而做标准评分卡等操作。从模型效果上来看，特征分箱主要是为了降低变量的复杂性，减少变量噪音对模型的影响，提高自变量和因变量的相关度。从而使模型更加稳定。


```python
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
```


```python
numerical_fea
```




    ['id',
     'loanAmnt',
     'term',
     'interestRate',
     'installment',
     'employmentTitle',
     'homeOwnership',
     'annualIncome',
     'verificationStatus',
     'isDefault',
     'purpose',
     'postCode',
     'regionCode',
     'dti',
     'delinquency_2years',
     'ficoRangeLow',
     'ficoRangeHigh',
     'openAcc',
     'pubRec',
     'pubRecBankruptcies',
     'revolBal',
     'revolUtil',
     'totalAcc',
     'initialListStatus',
     'applicationType',
     'title',
     'policyCode',
     'n0',
     'n1',
     'n2',
     'n2.1',
     'n4',
     'n5',
     'n6',
     'n7',
     'n8',
     'n9',
     'n10',
     'n11',
     'n12',
     'n13',
     'n14']




```python
category_fea
```




    ['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']




```python
data_train.grade
```




    0         E
    1         D
    2         D
    3         A
    4         C
             ..
    799995    C
    799996    A
    799997    C
    799998    A
    799999    B
    Name: grade, Length: 800000, dtype: object



数值型变量分析，数值型肯定是包括连续型变量和离散型变量的，找出来。

- 划分数值型变量中的连续变量和分类变量


```python
#过滤数值型类别特征
def get_numerical_serial_fea(data,feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea
numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(data_train,numerical_fea)
```


```python
numerical_serial_fea
```




    ['id',
     'loanAmnt',
     'interestRate',
     'installment',
     'employmentTitle',
     'annualIncome',
     'purpose',
     'postCode',
     'regionCode',
     'dti',
     'delinquency_2years',
     'ficoRangeLow',
     'ficoRangeHigh',
     'openAcc',
     'pubRec',
     'pubRecBankruptcies',
     'revolBal',
     'revolUtil',
     'totalAcc',
     'title',
     'n0',
     'n1',
     'n2',
     'n2.1',
     'n4',
     'n5',
     'n6',
     'n7',
     'n8',
     'n9',
     'n10',
     'n13',
     'n14']




```python
numerical_noserial_fea
```




    ['term',
     'homeOwnership',
     'verificationStatus',
     'isDefault',
     'initialListStatus',
     'applicationType',
     'policyCode',
     'n11',
     'n12']



- 数值类别型变量分析


```python
data_train['term'].value_counts()#离散型变量
```




    3    606902
    5    193098
    Name: term, dtype: int64




```python
data_train['homeOwnership'].value_counts()#离散型变量
```




    0    395732
    1    317660
    2     86309
    3       185
    5        81
    4        33
    Name: homeOwnership, dtype: int64




```python
data_train['verificationStatus'].value_counts()#离散型变量
```




    1    309810
    2    248968
    0    241222
    Name: verificationStatus, dtype: int64




```python
data_train['initialListStatus'].value_counts()#离散型变量
```




    0    466438
    1    333562
    Name: initialListStatus, dtype: int64




```python
data_train['applicationType'].value_counts()#离散型变量
```




    0    784586
    1     15414
    Name: applicationType, dtype: int64




```python
data_train['policyCode'].value_counts()#离散型变量，无用，全部一个值
```




    1.0    800000
    Name: policyCode, dtype: int64




```python
data_train['n11'].value_counts()#离散型变量，相差悬殊，用不用再分析
```




    0.0    729682
    1.0       540
    2.0        24
    4.0         1
    3.0         1
    Name: n11, dtype: int64




```python
data_train['n12'].value_counts()#离散型变量，相差悬殊，用不用再分析
```




    0.0    757315
    1.0      2281
    2.0       115
    3.0        16
    4.0         3
    Name: n12, dtype: int64



- 数值连续型变量分析


```python
#每个数字特征得分布可视化
f = pd.melt(data_train, value_vars=numerical_serial_fea)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```


![output_63_0.png](https://img-blog.csdnimg.cn/20200905092403482.png)


- 查看某一个数值型变量的分布，查看变量是否符合正态分布，如果不符合正太分布的变量可以log化后再观察下是否符合正态分布。
- 如果想统一处理一批数据变标准化 必须把这些之前已经正态化的数据提出


```python
#Ploting Transaction Amount Values Distribution
plt.figure(figsize=(16,12))
plt.suptitle('Transaction Values Distribution', fontsize=22)
plt.subplot(221)
sub_plot_1 = sns.distplot(data_train['loanAmnt'])
sub_plot_1.set_title("loanAmnt Distribuition", fontsize=18)
sub_plot_1.set_xlabel("")
sub_plot_1.set_ylabel("Probability", fontsize=15)

plt.subplot(222)
sub_plot_2 = sns.distplot(np.log(data_train['loanAmnt']))
sub_plot_2.set_title("loanAmnt (Log) Distribuition", fontsize=18)
sub_plot_2.set_xlabel("")
sub_plot_2.set_ylabel("Probability", fontsize=15)
```




    Text(0, 0.5, 'Probability')




![output_65_1.png](https://img-blog.csdnimg.cn/20200905092444845.png)


- 非数值类别型变量分析



```python
category_fea
```




    ['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']




```python
data_train['grade'].value_counts()
```




    B    233690
    C    227118
    A    139661
    D    119453
    E     55661
    F     19053
    G      5364
    Name: grade, dtype: int64




```python
data_train['subGrade'].value_counts()
```




    C1    50763
    B4    49516
    B5    48965
    B3    48600
    C2    47068
    C3    44751
    C4    44272
    B2    44227
    B1    42382
    C5    40264
    A5    38045
    A4    30928
    D1    30538
    D2    26528
    A1    25909
    D3    23410
    A3    22655
    A2    22124
    D4    21139
    D5    17838
    E1    14064
    E2    12746
    E3    10925
    E4     9273
    E5     8653
    F1     5925
    F2     4340
    F3     3577
    F4     2859
    F5     2352
    G1     1759
    G2     1231
    G3      978
    G4      751
    G5      645
    Name: subGrade, dtype: int64




```python
data_train['employmentLength'].value_counts()
```




    10+ years    262753
    2 years       72358
    < 1 year      64237
    3 years       64152
    1 year        52489
    5 years       50102
    4 years       47985
    6 years       37254
    8 years       36192
    7 years       35407
    9 years       30272
    Name: employmentLength, dtype: int64




```python
data_train['issueDate'].value_counts()
```




    2016-03-01    29066
    2015-10-01    25525
    2015-07-01    24496
    2015-12-01    23245
    2014-10-01    21461
                  ...  
    2007-08-01       23
    2007-07-01       21
    2008-09-01       19
    2007-09-01        7
    2007-06-01        1
    Name: issueDate, Length: 139, dtype: int64




```python
data_train['earliesCreditLine'].value_counts()
```




    Aug-2001    5567
    Sep-2003    5403
    Aug-2002    5403
    Oct-2001    5258
    Aug-2000    5246
                ... 
    May-1960       1
    Apr-1958       1
    Feb-1960       1
    Aug-1946       1
    Mar-1958       1
    Name: earliesCreditLine, Length: 720, dtype: int64




```python
data_train['isDefault'].value_counts()
```




    0    640390
    1    159610
    Name: isDefault, dtype: int64



总结：

- 上面我们用value_counts()等函数看了特征属性的分布，但是图表是概括原始信息最便捷的方式。
- 数无形时少直觉。
- 同一份数据集，在不同的尺度刻画上显示出来的图形反映的规律是不一样的。python将数据转化成图表，但结论是否正确需要由你保证。

### 2.3.6 变量分布可视化

#### 单一变量分布可视化


```python
plt.figure(figsize=(8, 8))
sns.barplot(data_train["employmentLength"].value_counts(dropna=False)[:20],
            data_train["employmentLength"].value_counts(dropna=False).keys()[:20])
plt.show()
```


![output_77_0.png](https://img-blog.csdnimg.cn/20200905092517742.png)


#### 根绝y值不同可视化x某个特征的分布

- 首先查看类别型变量在不同y值上的分布


```python
train_loan_fr = data_train.loc[data_train['isDefault'] == 1]
train_loan_nofr = data_train.loc[data_train['isDefault'] == 0]
```


```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_loan_fr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax1, title='Count of grade fraud')
train_loan_nofr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax2, title='Count of grade non-fraud')
train_loan_fr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax3, title='Count of employmentLength fraud')
train_loan_nofr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax4, title='Count of employmentLength non-fraud')
plt.show()
```


![output_81_0.png](https://img-blog.csdnimg.cn/20200905092545400.png)


- 其次查看连续型变量在不同y值上的分布


```python
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
data_train.loc[data_train['isDefault'] == 1] \
    ['loanAmnt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log Loan Amt - Fraud',
          color='r',
          xlim=(-3, 10),
         ax= ax1)
data_train.loc[data_train['isDefault'] == 0] \
    ['loanAmnt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log Loan Amt - Not Fraud',
          color='b',
          xlim=(-3, 10),
         ax=ax2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x126a44b50>




![output_83_1.png](https://img-blog.csdnimg.cn/20200905092739409.png)



```python
total = len(data_train)
total_amt = data_train.groupby(['isDefault'])['loanAmnt'].sum().sum()
plt.figure(figsize=(12,5))
plt.subplot(121)##1代表行，2代表列，所以一共有2个图，1代表此时绘制第一个图。
plot_tr = sns.countplot(x='isDefault',data=data_train)#data_train‘isDefault’这个特征每种类别的数量**
plot_tr.set_title("Fraud Loan Distribution \n 0: good user | 1: bad user", fontsize=14)
plot_tr.set_xlabel("Is fraud by count", fontsize=16)
plot_tr.set_ylabel('Count', fontsize=16)
for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 
    
percent_amt = (data_train.groupby(['isDefault'])['loanAmnt'].sum())
percent_amt = percent_amt.reset_index()
plt.subplot(122)
plot_tr_2 = sns.barplot(x='isDefault', y='loanAmnt',  dodge=True, data=percent_amt)
plot_tr_2.set_title("Total Amount in loanAmnt  \n 0: good user | 1: bad user", fontsize=14)
plot_tr_2.set_xlabel("Is fraud by percent", fontsize=16)
plot_tr_2.set_ylabel('Total Loan Amount Scalar', fontsize=16)
for p in plot_tr_2.patches:
    height = p.get_height()
    plot_tr_2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total_amt * 100),
            ha="center", fontsize=15)     
```


![output_84_0.png](https://img-blog.csdnimg.cn/20200905092816794.png)


### 2.3.7 时间格式数据处理及查看


```python
#转化成时间格式
data_train['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_train['issueDateDT'] = data_train['issueDate'].apply(lambda x: x-startdate).dt.days
```


```python
#转化成时间格式
data_test_a['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_test_a['issueDateDT'] = data_test_a['issueDate'].apply(lambda x: x-startdate).dt.days
```


```python
plt.hist(data_train['issueDateDT'], label='train');
plt.hist(data_test_a['issueDateDT'], label='test');
plt.legend();
plt.title('Distribution of issueDateDT dates');
#train 和 test issueDateDT 日期有重叠 所以使用基于时间的分割进行验证是不明智的
```


![output_88_0.png](https://img-blog.csdnimg.cn/20200905092845328.png)


### 2.3.8 掌握透视图可以让我们更好的了解数据


```python
#透视图 索引可以有多个，“columns（列）”是可选的，聚合函数aggfunc最后是被应用到了变量“values”中你所列举的项目上。
pivot = pd.pivot_table(data_train, index=['grade'], columns=['issueDateDT'], values=['loanAmnt'], aggfunc=np.sum)
```


```python
pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="21" halign="left">loanAmnt</th>
    </tr>
    <tr>
      <th>issueDateDT</th>
      <th>0</th>
      <th>30</th>
      <th>61</th>
      <th>92</th>
      <th>122</th>
      <th>153</th>
      <th>183</th>
      <th>214</th>
      <th>245</th>
      <th>274</th>
      <th>...</th>
      <th>3926</th>
      <th>3957</th>
      <th>3987</th>
      <th>4018</th>
      <th>4048</th>
      <th>4079</th>
      <th>4110</th>
      <th>4140</th>
      <th>4171</th>
      <th>4201</th>
    </tr>
    <tr>
      <th>grade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>NaN</td>
      <td>53650.0</td>
      <td>42000.0</td>
      <td>19500.0</td>
      <td>34425.0</td>
      <td>63950.0</td>
      <td>43500.0</td>
      <td>168825.0</td>
      <td>85600.0</td>
      <td>101825.0</td>
      <td>...</td>
      <td>13093850.0</td>
      <td>11757325.0</td>
      <td>11945975.0</td>
      <td>9144000.0</td>
      <td>7977650.0</td>
      <td>6888900.0</td>
      <td>5109800.0</td>
      <td>3919275.0</td>
      <td>2694025.0</td>
      <td>2245625.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>NaN</td>
      <td>13000.0</td>
      <td>24000.0</td>
      <td>32125.0</td>
      <td>7025.0</td>
      <td>95750.0</td>
      <td>164300.0</td>
      <td>303175.0</td>
      <td>434425.0</td>
      <td>538450.0</td>
      <td>...</td>
      <td>16863100.0</td>
      <td>17275175.0</td>
      <td>16217500.0</td>
      <td>11431350.0</td>
      <td>8967750.0</td>
      <td>7572725.0</td>
      <td>4884600.0</td>
      <td>4329400.0</td>
      <td>3922575.0</td>
      <td>3257100.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>NaN</td>
      <td>68750.0</td>
      <td>8175.0</td>
      <td>10000.0</td>
      <td>61800.0</td>
      <td>52550.0</td>
      <td>175375.0</td>
      <td>151100.0</td>
      <td>243725.0</td>
      <td>393150.0</td>
      <td>...</td>
      <td>17502375.0</td>
      <td>17471500.0</td>
      <td>16111225.0</td>
      <td>11973675.0</td>
      <td>10184450.0</td>
      <td>7765000.0</td>
      <td>5354450.0</td>
      <td>4552600.0</td>
      <td>2870050.0</td>
      <td>2246250.0</td>
    </tr>
    <tr>
      <th>D</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5500.0</td>
      <td>2850.0</td>
      <td>28625.0</td>
      <td>NaN</td>
      <td>167975.0</td>
      <td>171325.0</td>
      <td>192900.0</td>
      <td>269325.0</td>
      <td>...</td>
      <td>11403075.0</td>
      <td>10964150.0</td>
      <td>10747675.0</td>
      <td>7082050.0</td>
      <td>7189625.0</td>
      <td>5195700.0</td>
      <td>3455175.0</td>
      <td>3038500.0</td>
      <td>2452375.0</td>
      <td>1771750.0</td>
    </tr>
    <tr>
      <th>E</th>
      <td>7500.0</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>NaN</td>
      <td>17975.0</td>
      <td>1500.0</td>
      <td>94375.0</td>
      <td>116450.0</td>
      <td>42000.0</td>
      <td>139775.0</td>
      <td>...</td>
      <td>3983050.0</td>
      <td>3410125.0</td>
      <td>3107150.0</td>
      <td>2341825.0</td>
      <td>2225675.0</td>
      <td>1643675.0</td>
      <td>1091025.0</td>
      <td>1131625.0</td>
      <td>883950.0</td>
      <td>802425.0</td>
    </tr>
    <tr>
      <th>F</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>31250.0</td>
      <td>2125.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49000.0</td>
      <td>27000.0</td>
      <td>43000.0</td>
      <td>...</td>
      <td>1074175.0</td>
      <td>868925.0</td>
      <td>761675.0</td>
      <td>685325.0</td>
      <td>665750.0</td>
      <td>685200.0</td>
      <td>316700.0</td>
      <td>315075.0</td>
      <td>72300.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>G</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24625.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>56100.0</td>
      <td>243275.0</td>
      <td>224825.0</td>
      <td>64050.0</td>
      <td>198575.0</td>
      <td>245825.0</td>
      <td>53125.0</td>
      <td>23750.0</td>
      <td>25100.0</td>
      <td>1000.0</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 139 columns</p>
</div>



### 2.3.9 用pandas_profiling生成数据报告


```python
import pandas_profiling
```


```python
pfr = pandas_profiling.ProfileReport(data_train)
pfr.to_file("./example.html")
```

## 2.4 总结

数据探索性分析是我们初步了解数据，熟悉数据为特征工程做准备的阶段，甚至很多时候EDA阶段提取出来的特征可以直接当作规则来用。可见EDA的重要性，这个阶段的主要工作还是借助于各个简单的统计量来对数据整体的了解，分析各个类型变量相互之间的关系，以及用合适的图形可视化出来直观观察。希望本节内容能给初学者带来帮助，更期待各位学习者对其中的不足提出建议。

END.
【 言溪：Datawhale成员，金融风控爱好者。知乎地址：https://www.zhihu.com/people/exuding】

关于Datawhale：

Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。

本次数据挖掘路径学习，专题知识将在天池分享，详情可关注Datawhale：

![logo.png](https://img-blog.csdnimg.cn/2020090509294089.png)


