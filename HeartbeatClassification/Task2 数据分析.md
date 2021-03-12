# Task 2 数据分析

Tip: 此部分为零基础入门数据挖掘的 Task2 EDA-数据探索性分析 部分，带你来了解数据，熟悉数据，和数据做朋友，欢迎大家后续多多交流。

**赛题：心电图心跳信号多分类预测**

## 2.1 EDA 目标

- EDA的价值主要在于熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。
- 当了解了数据集之后我们下一步就是要去了解变量间的相互关系以及变量与预测值之间的存在关系。
- 引导数据科学从业者进行数据处理以及特征工程的步骤,使数据集的结构和特征集让接下来的预测问题更加可靠。
- 完成对于数据的探索性分析，并对于数据进行一些图表或者文字总结并打卡。

## 2.2 内容介绍

1. 载入各种数据科学以及可视化库:
   - 数据科学库 pandas、numpy、scipy；
   - 可视化库 matplotlib、seabon；
2. 载入数据：
   - 载入训练集和测试集；
   - 简略观察数据(head()+shape)；
3. 数据总览:
   - 通过describe()来熟悉数据的相关统计量
   - 通过info()来熟悉数据类型
4. 判断数据缺失和异常
   - 查看每列的存在nan情况
   - 异常值检测
5. 了解预测值的分布
   - 总体分布概况
   - 查看skewness and kurtosis
   - 查看预测值的具体频数

## 2.3 代码示例

#### 2.3.1 载入各种数据科学与可视化库

```python
#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')
import missingno as msno
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
```

#### 2.3.2 载入训练集和测试集

**导入训练集train.csv**

```python
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
Train_data = pd.read_csv('./train.csv')
```

**导入测试集testA.csv**

```python
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt 
Test_data = pd.read_csv('./testA.csv')
```

**所有特征集均脱敏处理(方便大家观看)**

- id - 心跳信号分配的唯一标识
- heartbeat_signals - 心跳信号序列
- label - 心跳信号类别（0、1、2、3）

`data.head().append(data.tail())`——观察首尾数据

`data.shape`——观察数据集的行列信息

**观察train首尾数据**

```python
Train_data.head().append(Train_data.tail())
```

```txt
<bound method DataFrame.info of           id                                  heartbeat_signals  label
0          0  0.9912297987616655,0.9435330436439665,0.764677...    0.0
1          1  0.9714822034884503,0.9289687459588268,0.572932...    0.0
2          2  1.0,0.9591487564065292,0.7013782792997189,0.23...    2.0
3          3  0.9757952826275774,0.9340884687738161,0.659636...    0.0
4          4  0.0,0.055816398940721094,0.26129357194994196,0...    2.0
...      ...                                                ...    ...
99995  99995  1.0,0.677705342021188,0.22239242747868546,0.25...    0.0
99996  99996  0.9268571578157265,0.9063471198026871,0.636993...    2.0
99997  99997  0.9258351628306013,0.5873839035878395,0.633226...    3.0
99998  99998  1.0,0.9947621698382489,0.8297017704865509,0.45...    2.0
99999  99999  0.9259994004527861,0.916476635326053,0.4042900...    0.0

[100000 rows x 3 columns]>
```

**观察train数据集的行列信息**

```python
Train_data.shape
```

```txt
(100000, 3)
```

**观察testA首尾数据**

```python
Test_data.head().append(Test_data.tail())
```

```txt
id	heartbeat_signals
0	100000	0.9915713654170097,1.0,0.6318163407681274,0.13...
1	100001	0.6075533139615096,0.5417083883163654,0.340694...
2	100002	0.9752726292239277,0.6710965234906665,0.686758...
3	100003	0.9956348033996116,0.9170249621481004,0.521096...
4	100004	1.0,0.8879490481178918,0.745564725322326,0.531...
19995	119995	1.0,0.8330283177934747,0.6340472606311671,0.63...
19996	119996	1.0,0.8259705825857048,0.4521053488322387,0.08...
19997	119997	0.951744840752379,0.9162611283848351,0.6675251...
19998	119998	0.9276692903808186,0.6771898159607004,0.242906...
19999	119999	0.6653212231837624,0.527064114047737,0.5166625...
```

**观察testA数据集的行列信**

```python
Test_data.shape
```

```txt
(20000, 2)
```

要养成看数据集的head()以及shape的习惯，这会让你每一步更放心，导致接下里的连串的错误, 如果对自己的pandas等操作不放心，建议执行一步看一下，这样会有效的方便你进行理解函数并进行操作

#### 2.3.3 总览数据概况

1. describe种有每列的统计量，个数count、平均值mean、方差std、最小值min、中位数25% 50% 75% 、以及最大值 看这个信息主要是瞬间掌握数据的大概的范围以及每个值的异常值的判断，比如有的时候会发现999 9999 -1 等值这些其实都是nan的另外一种表达方式，有的时候需要注意下
2. info 通过info来了解数据每列的type，有助于了解是否存在除了nan以外的特殊符号异常

`data.describe()`——获取数据的相关统计量

`data.info()`——获取数据类型

**获取train数据的相关统计量**

```python
Train_data.describe()
```

```txt
id	label
count	100000.000000	100000.000000
mean	49999.500000	0.856960
std	28867.657797	1.217084
min	0.000000	0.000000
25%	24999.750000	0.000000
50%	49999.500000	0.000000
75%	74999.250000	2.000000
max	99999.000000	3.000000
```

**获取train数据类型**

```python
Train_data.info
```

```txt
<bound method DataFrame.info of           id                              heartbeat_signals  label
0          0  0.9912297987616655,0.9435330436439665,0.764677...    0.0
1          1  0.9714822034884503,0.9289687459588268,0.572932...    0.0
2          2  1.0,0.9591487564065292,0.7013782792997189,0.23...    2.0
3          3  0.9757952826275774,0.9340884687738161,0.659636...    0.0
4          4  0.0,0.055816398940721094,0.26129357194994196,0...    2.0
...      ...                                                ...    ...
99995  99995  1.0,0.677705342021188,0.22239242747868546,0.25...    0.0
99996  99996  0.9268571578157265,0.9063471198026871,0.636993...    2.0
99997  99997  0.9258351628306013,0.5873839035878395,0.633226...    3.0
99998  99998  1.0,0.9947621698382489,0.8297017704865509,0.45...    2.0
99999  99999  0.9259994004527861,0.916476635326053,0.4042900...    0.0

[100000 rows x 3 columns]>
```

**获取testA数据的相关统计量**

```python
Test_data.describe()
```

```txt
 					id
count	20000.000000
mean	109999.500000
std	5773.647028
min	100000.000000
25%	104999.750000
50%	109999.500000
75%	114999.250000
max	119999.000000
```

**获取testA数据类型**

```python
Test_data.info()
```

```txt
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20000 entries, 0 to 19999
Data columns (total 2 columns):
 #   Column             Non-Null Count  Dtype 
---  ------             --------------  ----- 
 0   id                 20000 non-null  int64 
 1   heartbeat_signals  20000 non-null  object
dtypes: int64(1), object(1)
memory usage: 312.6+ KB
```

#### 2.3.4 判断数据缺失和异常

`data.isnull().sum()`——查看每列的存在nan情况

**查看trian每列的存在nan情况**

```python
Train_data.isnull().sum()
```

```python
id                   0
heartbeat_signals    0
label                0
dtype: int64
```

**查看testA每列的存在nan情况**

```python
Test_data.isnull().sum()
```

```python
id                   0
heartbeat_signals    0
dtype: int64
```

#### 2.3.5 了解预测值的分布

```python
Train_data['label']
```

```python
0        0.0
1        0.0
2        4.0
3        0.0
4        0.0
        ... 
99995    4.0
99996    0.0
99997    0.0
99998    0.0
99999    1.0
Name: label, Length: 100000, dtype: float64
```

```python
Train_data['label'].value_counts()
```

```python
0.0    58883
4.0    19660
2.0    12994
1.0     6522
3.0     1941
Name: label, dtype: int64
```

```python
## 1) 总体分布概况（无界约翰逊分布等）
import scipy.stats as st
y = Train_data['label']
plt.figure(1); plt.title('Default')
sns.distplot(y, rug=True, bins=20)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```

![image-20210131211440263](/Users/haoyue/Library/Application Support/typora-user-images/image-20210131211440263.png)

![image-20210131211452271](/Users/haoyue/Library/Application Support/typora-user-images/image-20210131211452271.png)

```python
# 2）查看skewness and kurtosis
sns.distplot(Train_data['label']);
print("Skewness: %f" % Train_data['label'].skew())
print("Kurtosis: %f" % Train_data['label'].kurt())
```

```python
Skewness: 0.917596
Kurtosis: -0.825276
```

![image-20210131211600245](/Users/haoyue/Library/Application Support/typora-user-images/image-20210131211600245.png)

```python
Train_data.skew(), Train_data.kurt()
```

```python
(id       0.000000
 label    0.917596
 dtype: float64, id      -1.200000
 label   -0.825276
 dtype: float64)
```

```python
sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness')
```

![image-20210131211722579](/Users/haoyue/Library/Application Support/typora-user-images/image-20210131211722579.png)

```python
## 3) 查看预测值的具体频数
plt.hist(Train_data['label'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()
```

![image-20210131211751810](/Users/haoyue/Library/Application Support/typora-user-images/image-20210131211751810.png)



#### 2.3.7 用pandas_profiling生成数据报告

```python
import pandas_profiling
```

```python
pfr = pandas_profiling.ProfileReport(data_train)
pfr.to_file("./example.html")
```



### 2.4 总结

数据探索性分析是我们初步了解数据，熟悉数据为特征工程做准备的阶段，甚至很多时候EDA阶段提取出来的特征可以直接当作规则来用。可见EDA的重要性，这个阶段的主要工作还是借助于各个简单的统计量来对数据整体的了解，分析各个类型变量相互之间的关系，以及用合适的图形可视化出来直观观察。希望本节内容能给初学者带来帮助，更期待各位学习者对其中的不足提出建议。



**关于Datawhale：**

> Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。

本次数据挖掘路径学习，专题知识将在天池分享，详情可关注Datawhale：

![img](https://camo.githubusercontent.com/20948c03ba006b30749edb1c2eb5c73c59f82c9c6538d4cab8278ad700dee5be/687474703a2f2f6a75707465722d6f73732e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f7075626c69632f66696c65732f696d6167652f323332363534313034322f313538343432363332363932305f39464f554578473262652e6a7067)