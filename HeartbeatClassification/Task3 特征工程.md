# Task3 特征工程

此部分为零基础入门数据挖掘-心跳信号分类预测的 Task3 特征工程部分，带你来了解时间序列特征工程以及分析方法，欢迎大家后续多多交流。

赛题：零基础入门数据挖掘-心跳信号分类预测

项目地址：
比赛地址：

## 3.1 学习目标

* 学习时间序列数据的特征预处理方法
* 学习时间序列特征处理工具 Tsfresh（TimeSeries Fresh）的使用

## 3.2 内容介绍
* 数据预处理
	* 时间序列数据格式处理
	* 加入时间步特征time
* 特征工程
	* 时间序列特征构造
	* 特征筛选
	* 使用 tsfresh 进行时间序列特征处理

## 3.3 代码示例

### 3.3.1 导入包并读取数据

```python
# 包导入
import pandas as pd
import numpy as np
import tsfresh as tsf
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
```

```python
# 数据读取
data_train = pd.read_csv("train.csv")
data_test_A = pd.read_csv("testA.csv")

print(data_train.shape)
print(data_test_A.shape)
```

```python
(100000, 3)
(20000, 2)
```

```python
data_train.head()
```

```python
  	id		heartbeat_signals	                                label
0		0			0.9912297987616655,0.9435330436439665,0.764677…		0.0
1		1			0.9714822034884503,0.9289687459588268,0.572932…		0.0
2		2			1.0,0.9591487564065292,0.7013782792997189,0.23…		2.0
3		3			0.9757952826275774,0.9340884687738161,0.659636…		0.0
4		4			0.0,0.055816398940721094,0.26129357194994196,0…		2.0
```

```python
data_test_A.head()
```

```python
		id	    heartbeat_signals
0		100000	0.9915713654170097,1.0,0.6318163407681274,0.13…
1		100001	0.6075533139615096,0.5417083883163654,0.340694…
2		100002	0.9752726292239277,0.6710965234906665,0.686758…
3		100003	0.9956348033996116,0.9170249621481004,0.521096…
4		100004	1.0,0.8879490481178918,0.745564725322326,0.531…
```


### 3.3.2 数据预处理
```python
# 对心电特征进行行转列处理，同时为每个心电信号加入时间步特征time
train_heartbeat_df = data_train["heartbeat_signals"].str.split(",", expand=True).stack()
train_heartbeat_df = train_heartbeat_df.reset_index()
train_heartbeat_df = train_heartbeat_df.set_index("level_0")
train_heartbeat_df.index.name = None
train_heartbeat_df.rename(columns={"level_1":"time", 0:"heartbeat_signals"}, inplace=True)
train_heartbeat_df["heartbeat_signals"] = train_heartbeat_df["heartbeat_signals"].astype(float)

train_heartbeat_df
```

```python
			time		heartbeat_signals
0			0				0.991230
0			1				0.943533
0			2				0.764677
0			3				0.618571
0			4				0.379632
...		...			...
99999	200			0.000000
99999	201			0.000000
99999	202			0.000000
99999	203			0.000000
99999	204			0.000000

20500000 rows × 2 columns
```

```python
# 将处理后的心电特征加入到训练数据中，同时将训练数据label列单独存储
data_train_label = data_train["label"]
data_train = data_train.drop("label", axis=1)
data_train = data_train.drop("heartbeat_signals", axis=1)
data_train = data_train.join(train_heartbeat_df)

data_train
```

```python
			id		time	heartbeat_signals
0			0			0			0.991230
0			0			1			0.943533
0			0			2			0.764677
0			0			3			0.618571
0			0			4			0.379632
...		...		...		...
99999	99999	200		0.0
99999	99999	201		0.0
99999	99999	202		0.0
99999	99999	203		0.0
99999	99999	204		0.0

20500000 rows × 4 columns
```

```python
data_train[data_train["id"]==1]
```

```python
			id		time	heartbeat_signals
1			1			0			0.971482
1			1			1			0.928969
1			1			2			0.572933
1			1			3			0.178457
1			1			4			0.122962
...		...		...		...
1			1			200		0.0
1			1			201		0.0
1			1			202		0.0
1			1			203		0.0
1			1			204		0.0

205 rows × 4 columns
```

可以看到，每个样本的心电特征都由205个时间步的心电信号组成。


### 3.3.3 使用 tsfresh 进行时间序列特征处理
1. 特征抽取
**Tsfresh（TimeSeries Fresh）**是一个Python第三方工具包。 它可以自动计算大量的时间序列数据的特征。此外，该包还包含了特征重要性评估、特征选择的方法，因此，不管是基于时序数据的分类问题还是回归问题，tsfresh都会是特征提取一个不错的选择。官方文档：[Introduction — tsfresh 0.17.1.dev24+g860c4e1 documentation](https://tsfresh.readthedocs.io/en/latest/text/introduction.html)
```python
from tsfresh import extract_features

# 特征提取
train_features = extract_features(data_train, column_id='id', column_sort='time')
train_features
```

```python
id		sum_values		abs_energy		mean_abs_change		mean_change 	...
0			38.927945			18.216197			0.019894					-0.004859			...
1			19.445634			7.705092			0.019952					-0.004762			...
2			21.192974			9.140423			0.009863					-0.004902			...
...		...						...						...								...						...
99997	40.897057			16.412857			0.019470					-0.004538			...
99998	42.333303			14.281281			0.017032					-0.004902			...
99999	53.290117			21.637471			0.021870					-0.004539			...

100000 rows × 779 columns
```




2. 特征选择 
train_features中包含了heartbeat_signals的779种常见的时间序列特征（所有这些特征的解释可以去看官方文档），这其中有的特征可能为NaN值（产生原因为当前数据不支持此类特征的计算），使用以下方式去除NaN值：
```python
from tsfresh.utilities.dataframe_functions import impute

# 去除抽取特征中的NaN值
impute(train_features)
```

```python
id		sum_values		abs_energy		mean_abs_change		mean_change 	...
0			38.927945			18.216197			0.019894					-0.004859			...
1			19.445634			7.705092			0.019952					-0.004762			...
2			21.192974			9.140423			0.009863					-0.004902			...
...		...						...						...								...						...
99997	40.897057			16.412857			0.019470					-0.004538			...
99998	42.333303			14.281281			0.017032					-0.004902			...
99999	53.290117			21.637471			0.021870					-0.004539			...

100000 rows × 779 columns
```

接下来，按照特征和响应变量之间的相关性进行特征选择，这一过程包含两步：首先单独计算每个特征和响应变量之间的相关性，然后利用Benjamini-Yekutieli procedure [1] 进行特征选择，决定哪些特征可以被保留。
```python
from tsfresh import select_features

# 按照特征和数据label之间的相关性进行特征选择
train_features_filtered = select_features(train_features, data_train_label)

train_features_filtered
```

```python
id		sum_values		fft_coefficient__attr_"abs"__coeff_35		fft_coefficient__attr_"abs"__coeff_34		...
0			38.927945			1.168685																0.982133																...
1			19.445634			1.460752																1.924501																...
2			21.192974			1.787166																2.1469872																...
...		...						...																			...																			...
99997	40.897057			1.190514																0.674603																...
99998	42.333303			1.237608																1.325212																...
99999	53.290117			0.154759																2.921164																...

100000 rows × 700 columns
```

可以看到经过特征选择，留下了700个特征。

## References

[1] Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. Annals of statistics, 1165–1188



**Task3 特征工程 END.**

--- By: 吉米杜

> 平安NLP算法工程师，Datawhale成员，Coggle开源小组成员
>
> 博客：https://blog.csdn.net/duxiaodong1122?spm=1011.2124.3001.5343&type=blog

关于Datawhale：
Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
本次数据挖掘路径学习，专题知识将在天池分享，详情可关注Datawhale：

![logo.png](https://img-blog.csdnimg.cn/2020091301022698.png#pic_center)