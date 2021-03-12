# Task 5: 模型融合

此部分为零基础入门数据挖掘之心电图分类的 Task5 建模融合部分，带你来了解各种模型融合方法及策略，欢迎大家后续多多交流。

赛题：零基础入门数据挖掘 - 心电图分类预测

项目地址：

比赛地址：


## 5.1 学习目标

- 学习融合策略
- 完成相应学习打卡任务 

## 5.2 内容介绍

https://mlwave.com/kaggle-ensembling-guide/  
https://github.com/MLWave/Kaggle-Ensemble-Guide

模型融合是比赛后期一个重要的环节，大体来说有如下的类型方式。

1. 简单加权融合:
    - 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）；
    - 分类：投票（Voting)
    - 综合：排序融合(Rank averaging)，log融合

2. stacking/blending:
    - 构建多层模型，并利用预测结果再拟合预测。

3. boosting/bagging（在xgboost，Adaboost,GBDT中已经用到）:
    - 多树的提升方法

## 5.3 相关理论介绍

stacking具体原理详解
1. https://www.cnblogs.com/yumoye/p/11024137.html
2. https://zhuanlan.zhihu.com/p/26890738


## 5.4 代码实例

### 5.4.1 回归\分类概率-融合：

#### (1) 简单加权平均，结果直接融合


```python
import numpy as np
import pandas as pd
from sklearn import metrics

## 生成一些简单的样本数据，test_prei 代表第i个模型的预测值
test_pre1 = [1.2, 3.2, 2.1, 6.2]
test_pre2 = [0.9, 3.1, 2.0, 5.9]
test_pre3 = [1.1, 2.9, 2.2, 6.0]

# y_test_true 代表第模型的真实值
y_test_true = [1, 3, 2, 6] 

## 定义结果的加权平均函数
def Weighted_method(test_pre1,test_pre2,test_pre3,w=[1/3,1/3,1/3]):
    Weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)+w[2]*pd.Series(test_pre3)
    return Weighted_result

# 各模型的预测结果计算MAE
print('Pred1 MAE:',metrics.mean_absolute_error(y_test_true, test_pre1))
print('Pred2 MAE:',metrics.mean_absolute_error(y_test_true, test_pre2))
print('Pred3 MAE:',metrics.mean_absolute_error(y_test_true, test_pre3))

## 根据加权计算MAE
w = [0.3,0.4,0.3] # 定义比重权值
Weighted_pre = Weighted_method(test_pre1,test_pre2,test_pre3,w)
print('Weighted_pre MAE:',metrics.mean_absolute_error(y_test_true, Weighted_pre))
```

    Pred1 MAE: 0.1750000000000001
    Pred2 MAE: 0.07499999999999993
    Pred3 MAE: 0.10000000000000009
    Weighted_pre MAE: 0.05750000000000027


可以发现加权结果相对于之前的结果是有提升的，这种我们称其为简单的加权平均。  
还有一些特殊的形式，比如mean平均，median平均


```python
## 定义结果的加权平均函数
def Mean_method(test_pre1,test_pre2,test_pre3):
    Mean_result = pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).mean(axis=1)
    return Mean_result

Mean_pre = Mean_method(test_pre1,test_pre2,test_pre3)
print('Mean_pre MAE:',metrics.mean_absolute_error(y_test_true, Mean_pre))

## 定义结果的加权平均函数
def Median_method(test_pre1,test_pre2,test_pre3):
    Median_result = pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).median(axis=1)
    return Median_result

Median_pre = Median_method(test_pre1,test_pre2,test_pre3)
print('Median_pre MAE:',metrics.mean_absolute_error(y_test_true, Median_pre))
```

    Mean_pre MAE: 0.06666666666666693
    Median_pre MAE: 0.07500000000000007


#### (2) Stacking融合(回归)


```python
from sklearn import linear_model

def Stacking_method(train_reg1,train_reg2,train_reg3,y_train_true,test_pre1,test_pre2,test_pre3,model_L2= linear_model.LinearRegression()):
    model_L2.fit(pd.concat([pd.Series(train_reg1),pd.Series(train_reg2),pd.Series(train_reg3)],axis=1).values,y_train_true)
    Stacking_result = model_L2.predict(pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).values)
    return Stacking_result

## 生成一些简单的样本数据，test_prei 代表第i个模型的预测值
train_reg1 = [3.2, 8.2, 9.1, 5.2]
train_reg2 = [2.9, 8.1, 9.0, 4.9]
train_reg3 = [3.1, 7.9, 9.2, 5.0]
# y_test_true 代表第模型的真实值
y_train_true = [3, 8, 9, 5] 

test_pre1 = [1.2, 3.2, 2.1, 6.2]
test_pre2 = [0.9, 3.1, 2.0, 5.9]
test_pre3 = [1.1, 2.9, 2.2, 6.0]

# y_test_true 代表第模型的真实值
y_test_true = [1, 3, 2, 6] 

model_L2= linear_model.LinearRegression()
Stacking_pre = Stacking_method(train_reg1,train_reg2,train_reg3,y_train_true,
                               test_pre1,test_pre2,test_pre3,model_L2)
print('Stacking_pre MAE:',metrics.mean_absolute_error(y_test_true, Stacking_pre))
```

    Stacking_pre MAE: 0.04213483146067404


可以发现模型结果相对于之前有进一步的提升，这是我们需要注意的一点是，对于第二层Stacking的模型不宜选取的过于复杂，这样会导致模型在训练集上过拟合，从而使得在测试集上并不能达到很好的效果。

### 5.4.2 分类模型融合


```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
```

#### (1) Voting投票机制

Voting即投票机制，分为软投票和硬投票两种，其原理采用少数服从多数的思想。


```python
'''
硬投票：对多个模型直接进行投票，不区分模型结果的相对重要度，最终投票数最多的类为最终被预测的类。
'''
iris = datasets.load_iris()

x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

clf1 = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, min_child_weight=2, subsample=0.7,
                     colsample_bytree=0.6, objective='binary:logistic')
clf2 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10,
                              min_samples_leaf=63,oob_score=True)
clf3 = SVC(C=0.1)

# 硬投票
eclf = VotingClassifier(estimators=[('lgb', clf1), ('rf', clf2), ('svc', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['LGB', 'Random Forest', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
```

    Accuracy: 0.95 (+/- 0.05) [LGB]
    Accuracy: 0.33 (+/- 0.00) [Random Forest]
    Accuracy: 0.92 (+/- 0.03) [SVM]
    Accuracy: 0.95 (+/- 0.05) [Ensemble]


#### (2) 分类的Stacking\Blending融合：

stacking是一种分层模型集成框架。

以两层为例，第一层由多个基学习器组成，其输入为原始训练集，第二层的模型则是以第一层基学习器的输出作为训练集进行再训练，从而得到完整的stacking模型, stacking两层模型都使用了全部的训练数据。


```python
'''
5-Fold Stacking
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
import pandas as pd
#创建训练的数据集
data_0 = iris.data
data = data_0[:100,:]

target_0 = iris.target
target = target_0[:100]

#模型融合中使用到的各个单模型
clfs = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]
 
#切分一部分数据作为测试集
X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.3, random_state=2020)

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

#5折stacking
n_splits = 5
skf = StratifiedKFold(n_splits)
skf = skf.split(X, y)

for j, clf in enumerate(clfs):
    #依次训练各个单模型
    dataset_blend_test_j = np.zeros((X_predict.shape[0], 5))
    for i, (train, test) in enumerate(skf):
        #5-Fold交叉训练，使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    #对于测试集，直接用这k个模型的预测值均值作为新的特征。
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))

clf = LogisticRegression(solver='lbfgs')
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

print("Val auc Score of Stacking: %f" % (roc_auc_score(y_predict, y_submission)))
```

    val auc Score: 1.000000
    val auc Score: 0.500000
    val auc Score: 0.500000
    val auc Score: 0.500000
    val auc Score: 0.500000
    Val auc Score of Stacking: 1.000000


Blending，其实和Stacking是一种类似的多层模型融合的形式  
- 其主要思路是把原始的训练集先分成两部分，比如70%的数据作为新的训练集，剩下30%的数据作为测试集。
- 在第一层，我们在这70%的数据上训练多个模型，然后去预测那30%数据的label，同时也预测test集的label。
- 在第二层，我们就直接用这30%数据在第一层预测的结果做为新特征继续训练，然后用test集第一层预测的label做特征，用第二层训练的模型做进一步预测

其优点在于
- 比stacking简单（因为不用进行k次的交叉验证来获得stacker feature）
- 避开了一个信息泄露问题：generlizers和stacker使用了不一样的数据集

缺点在于：
- 使用了很少的数据（第二阶段的blender只使用training set10%的量）
- blender可能会过拟合
- stacking使用多次的交叉验证会比较稳健


```python
'''
Blending
'''
 
#创建训练的数据集
#创建训练的数据集
data_0 = iris.data
data = data_0[:100,:]

target_0 = iris.target
target = target_0[:100]
 
#模型融合中使用到的各个单模型
clfs = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        #ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

#切分一部分数据作为测试集
X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.3, random_state=2020)

#切分训练数据集为d1,d2两部分
X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.5, random_state=2020)
dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))
 
for j, clf in enumerate(clfs):
    #依次训练各个单模型
    clf.fit(X_d1, y_d1)
    y_submission = clf.predict_proba(X_d2)[:, 1]
    dataset_d1[:, j] = y_submission
    #对于测试集，直接用这k个模型的预测值作为新的特征。
    dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_d2[:, j]))

#融合使用的模型
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_d1, y_d2)
y_submission = clf.predict_proba(dataset_d2)[:, 1]
print("Val auc Score of Blending: %f" % (roc_auc_score(y_predict, y_submission)))
```

    val auc Score: 1.000000
    val auc Score: 1.000000
    val auc Score: 1.000000
    val auc Score: 1.000000
    val auc Score: 1.000000
    Val auc Score of Blending: 1.000000


### 5.4.3 一些其它方法

将特征放进模型中预测，并将预测结果变换并作为新的特征加入原有特征中再经过模型预测结果 （Stacking变化）  
（可以反复预测多次将结果加入最后的特征中）


```python
def Ensemble_add_feature(train,test,target,clfs):
    
    # n_flods = 5
    # skf = list(StratifiedKFold(y, n_folds=n_flods))

    train_ = np.zeros((train.shape[0],len(clfs*2)))
    test_ = np.zeros((test.shape[0],len(clfs*2)))

    for j,clf in enumerate(clfs):
        '''依次训练各个单模型'''
        # print(j, clf)
        '''使用第1个部分作为预测，第2部分来训练模型，获得其预测的输出作为第2部分的新特征。'''
        # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]

        clf.fit(train,target)
        y_train = clf.predict(train)
        y_test = clf.predict(test)

        ## 新特征生成
        train_[:,j*2] = y_train**2
        test_[:,j*2] = y_test**2
        train_[:, j+1] = np.exp(y_train)
        test_[:, j+1] = np.exp(y_test)
        # print("val auc Score: %f" % r2_score(y_predict, dataset_d2[:, j]))
        print('Method ',j)

    train_ = pd.DataFrame(train_)
    test_ = pd.DataFrame(test_)
    return train_,test_

```


```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

data_0 = iris.data
data = data_0[:100,:]

target_0 = iris.target
target = target_0[:100]

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3)
x_train = pd.DataFrame(x_train) ; x_test = pd.DataFrame(x_test)

#模型融合中使用到的各个单模型
clfs = [LogisticRegression(),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

New_train,New_test = Ensemble_add_feature(x_train,x_test,y_train,clfs)

clf = LogisticRegression()
# clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(New_train, y_train)
y_emb = clf.predict_proba(New_test)[:, 1]

print("Val auc Score of stacking: %f" % (roc_auc_score(y_test, y_emb)))
```

    Method  0
    Method  1
    Method  2
    Method  3
    Method  4
    Val auc Score of stacking: 1.000000


## 5.5 本赛题示例

### 5.5.1 准备工作

准备工作进行内容有：
1. 导入数据集并进行简单的预处理
2. 将数据集划分成训练集和验证集
3. 构建单模：Random Forest，LGB，NN
4. 读取并演示如何利用融合模型生成可提交预测数据


```python
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
%matplotlib inline

import itertools
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
# from mlxtend.plotting import plot_learning_curves
# from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

这里引入一个降内存的函数。


```python
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
```


```python
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/testA.csv')

# 简单预处理
train_list = []
for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
    
test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

train = pd.DataFrame(np.array(train_list))
test = pd.DataFrame(np.array(test_list))

# id列不算入特征
features = ['s_'+str(i) for i in range(len(train_list[0])-2)] 
train.columns = ['id'] + features + ['label']
test.columns = ['id'] + features

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
```

    Memory usage of dataframe is 157.93 MB
    Memory usage after optimization is: 39.67 MB
    Decreased by 74.9%
    Memory usage of dataframe is 31.43 MB
    Memory usage after optimization is: 7.90 MB
    Decreased by 74.9%



```python
# 根据8：2划分训练集和校验集
X_train = train.drop(['id','label'], axis=1)
y_train = train['label']

# 测试集
X_test = test.drop(['id'], axis=1)

# 第一次运行可以先用一个subdata，这样速度会快些
X_train = X_train.iloc[:50000,:20]
y_train = y_train.iloc[:50000]
X_test = X_test.iloc[:,:20]

# 划分训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
```


```python
# 单模函数
def build_model_rf(X_train,y_train):
    model = RandomForestRegressor(n_estimators = 100)
    model.fit(X_train, y_train)
    return model


def build_model_lgb(X_train,y_train):
    model = lgb.LGBMRegressor(num_leaves=63,learning_rate = 0.1,n_estimators = 100)
    model.fit(X_train, y_train)
    return model


def build_model_nn(X_train,y_train):
    model = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')
    model.fit(X_train, y_train)
    return model
```


```python
# 这里针对三个单模进行训练，其中subA_rf/lgb/nn都是可以提交的模型
# 单模没有进行调参，因此是弱分类器，效果可能不是很好。

print('predict rf...')
model_rf = build_model_rf(X_train,y_train)
val_rf = model_rf.predict(X_val)
subA_rf = model_rf.predict(X_test)


print('predict lgb...')
model_lgb = build_model_lgb(X_train,y_train)
val_lgb = model_lgb.predict(X_val)
subA_lgb = model_rf.predict(X_test)


print('predict NN...')
model_nn = build_model_nn(X_train,y_train)
val_nn = model_nn.predict(X_val)
subA_nn = model_rf.predict(X_test)
```

    predict rf...
    predict lgb...
    predict NN...


### 5.5.2 加权融合

首先我们尝试加权融合模型：
- 如果没有给权重矩阵，就是均值融合模型
- 权重矩阵可以进行自定义，这里我们是用三个单模进行融合。如果有更多需要更改矩阵size


```python
# 加权融合模型，如果w没有变，就是均值融合
def Weighted_method(test_pre1,test_pre2,test_pre3,w=[1/3,1/3,1/3]):
    Weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)+w[2]*pd.Series(test_pre3)
    return Weighted_result

# 初始权重，可以进行自定义，这里我们随便设置一个权重
w = [0.2, 0.3, 0.5]

val_pre = Weighted_method(val_rf,val_lgb,val_nn,w)
MAE_Weighted = mean_absolute_error(y_val,val_pre)
print('MAE of Weighted of val:',MAE_Weighted)
```

    MAE of Weighted of val: 0.09326


这里单独展示一下将多个单模预测结果融合成融和模型结果


```python
## 预测数据部分
subA = Weighted_method(subA_rf,subA_lgb,subA_nn,w)

## 生成提交文件
sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = subA
sub.to_csv('./sub_Weighted.csv',index=False)
```

### 5.5.3 Stacking融合


```python
## Stacking

## 第一层
train_rf_pred = model_rf.predict(X_train)
train_lgb_pred = model_lgb.predict(X_train)
train_nn_pred = model_nn.predict(X_train)

stacking_X_train = pd.DataFrame()
stacking_X_train['Method_1'] = train_rf_pred
stacking_X_train['Method_2'] = train_lgb_pred
stacking_X_train['Method_3'] = train_nn_pred

stacking_X_val = pd.DataFrame()
stacking_X_val['Method_1'] = val_rf
stacking_X_val['Method_2'] = val_lgb
stacking_X_val['Method_3'] = val_nn

stacking_X_test = pd.DataFrame()
stacking_X_test['Method_1'] = subA_rf
stacking_X_test['Method_2'] = subA_lgb
stacking_X_test['Method_3'] = subA_nn
```


```python
stacking_X_test.head()
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
      <th>Method_1</th>
      <th>Method_2</th>
      <th>Method_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 第二层是用random forest
model_lr_stacking = build_model_rf(stacking_X_train,y_train)

## 训练集
train_pre_Stacking = model_lr_stacking.predict(stacking_X_train)
print('MAE of stacking:',mean_absolute_error(y_train,train_pre_Stacking))

## 验证集
val_pre_Stacking = model_lr_stacking.predict(stacking_X_val)
print('MAE of stacking:',mean_absolute_error(y_val,val_pre_Stacking))

## 预测集
print('Predict stacking...')
subA_Stacking = model_lr_stacking.predict(stacking_X_test)

```

    MAE of stacking: 0.0
    MAE of stacking: 0.03384
    Predict stacking...


## 5.6 经验总结

模型融合是数据挖掘比赛后期上分的主要方式，尤其是进行队伍合并后，模型融合有很多优势。总结一下三个方面：
1. **结果层面的融合**，这种是最常见的融合方法，其可行的融合方法也有很多，比如根据结果的得分进行加权融合，还可以做Log，exp处理等。在做结果融合的时候。有一个很重要的条件是模型结果的得分要比较近似但结果的差异要比较大，这样的结果融合往往有比较好的效果提升。如果不满足这个条件带来的效果很低，甚至是负效果。

2. **特征层面的融合**，这个层面叫融合融合并不准确，主要是队伍合并后大家可以相互学习特征工程。如果我们用同种模型训练，可以把特征进行切分给不同的模型，然后在后面进行模型或者结果融合有时也能产生比较好的效果。

3. **模型层面的融合**，模型层面的融合可能就涉及模型的堆叠和设计，比如加stacking，部分模型的结果作为特征输入等，这些就需要多实验和思考了，基于模型层面的融合最好不同模型类型要有一定的差异，用同种模型不同的参数的收益一般是比较小的。


**END.**

【 张晋 ：Datawhale成员，算法竞赛爱好者。CSDN：https://blog.csdn.net/weixin_44585839/】



关于Datawhale：

> Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。

本次数据挖掘路径学习，专题知识将在天池分享，详情可关注Datawhale：

![logo.png](https://img-blog.csdnimg.cn/2020090509294089.png)
