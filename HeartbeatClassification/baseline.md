

# baseline

## 1.导入第三方包

```python
import os
import gc
import math

import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
```

## 2.读取数据

```python
train = pd.read_csv('train.csv')
test=pd.read_csv('testA.csv')
train.head()
```

| id   | heartbeat_signals                                 | label |
| ---- | ------------------------------------------------- | ----- |
| 0    | 0.9912297987616655,0.9435330436439665,0.764677... | 0.0   |
| 1    | 0.9912297987616655,0.9435330436439665,0.764677... | 0.0   |
| 2    | 1.0,0.9591487564065292,0.7013782792997189,0.23... | 2.0   |
| 3    | 0.9757952826275774,0.9340884687738161,0.659636... | 0.0   |
| 4    | 0.0,0.055816398940721094,0.26129357194994196,0... | 2.0   |

```
test.head()
```

| id     | hearbeat_signals                                  |
| ------ | ------------------------------------------------- |
| 100000 | 0.9915713654170097,1.0,0.6318163407681274,0.13... |
| 100001 | 0.6075533139615096,0.5417083883163654,0.340694... |
| 100002 | 0.9752726292239277,0.6710965234906665,0.686758... |
| 100003 | 0.9956348033996116,0.9170249621481004,0.521096... |
| 100004 | 1.0,0.8879490481178918,0.745564725322326,0.531... |

## 3.数据预处理

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
# 简单预处理
train_list = []

for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

train = pd.DataFrame(np.array(train_list))
train.columns = ['id'] + ['s_'+str(i) for i in range(len(train_list[0])-2)] + ['label']
train = reduce_mem_usage(train)

test_list=[]
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_'+str(i) for i in range(len(test_list[0])-1)]
test = reduce_mem_usage(test)
```

```
Memory usage of dataframe is 157.93 MB
Memory usage after optimization is: 39.67 MB
Decreased by 74.9%
Memory usage of dataframe is 31.43 MB
Memory usage after optimization is: 7.90 MB
Decreased by 74.9%
```

## 4.训练数据/测试数据准备

```python
x_train = train.drop(['id','label'], axis=1)
y_train = train['label']
x_test=test.drop(['id'], axis=1)
```

## 5.模型训练

```python
def abs_sum(y_pre,y_tru):
    y_pre=np.array(y_pre)
    y_tru=np.array(y_tru)
    loss=sum(sum(abs(y_pre-y_tru)))
    return loss
```

```python
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test = np.zeros((test_x.shape[0],4))

    cv_scores = []
    onehot_encoder = OneHotEncoder(sparse=False)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'num_leaves': 2 ** 5,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': seed,
                'nthread': 28,
                'n_jobs':24,
                'verbose': -1,
            }

            model = clf.train(params, 
                      train_set=train_matrix, 
                      valid_sets=valid_matrix, 
                      num_boost_round=2000, 
                      verbose_eval=100, 
                      early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration) 
            
        val_y=np.array(val_y).reshape(-1, 1)
        val_y = onehot_encoder.fit_transform(val_y)
        print('预测的概率矩阵为：')
        print(test_pred)
        test += test_pred
        score=abs_sum(val_y, val_pred)
        cv_scores.append(score)
        print(cv_scores)
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    test=test/kf.n_splits

    return test
```

```python
def lgb_model(x_train, y_train, x_test):
    lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_test
lgb_test = lgb_model(x_train, y_train, x_test)
```

```
************************************ 1 ************************************
[LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
Training until validation scores don't improve for 200 rounds
[100]	valid_0's multi_logloss: 0.0525735
[200]	valid_0's multi_logloss: 0.0422444
[300]	valid_0's multi_logloss: 0.0407076
[400]	valid_0's multi_logloss: 0.0420398
Early stopping, best iteration is:
[289]	valid_0's multi_logloss: 0.0405457
预测的概率矩阵为：
[[9.99969791e-01 2.85197261e-05 1.00341946e-06 6.85357631e-07]
 [7.93287264e-05 7.69060914e-04 9.99151590e-01 2.00810971e-08]
 [5.75356884e-07 5.04051497e-08 3.15322414e-07 9.99999059e-01]
 ...
 [6.79267940e-02 4.30206297e-04 9.31640185e-01 2.81516302e-06]
 [9.99960477e-01 3.94098074e-05 8.34030725e-08 2.94638661e-08]
 [9.88705846e-01 2.14081630e-03 6.67418381e-03 2.47915423e-03]]
[607.0736049372186]
************************************ 2 ************************************
[LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
Training until validation scores don't improve for 200 rounds
[100]	valid_0's multi_logloss: 0.0566626
[200]	valid_0's multi_logloss: 0.0450852
[300]	valid_0's multi_logloss: 0.044078
[400]	valid_0's multi_logloss: 0.0455546
Early stopping, best iteration is:
[275]	valid_0's multi_logloss: 0.0437793
预测的概率矩阵为：
[[9.99991401e-01 7.69109547e-06 6.65504756e-07 2.42084688e-07]
 [5.72380482e-05 1.32812809e-03 9.98614607e-01 2.66534396e-08]
 [2.82123411e-06 4.13195205e-07 1.34026965e-06 9.99995425e-01]
 ...
 [6.96398024e-02 6.52459907e-04 9.29685742e-01 2.19960932e-05]
 [9.99972366e-01 2.75069005e-05 7.68142933e-08 5.07415018e-08]
 [9.67263676e-01 7.26154408e-03 2.41533542e-02 1.32142531e-03]]
[607.0736049372186, 623.4313863731124]
************************************ 3 ************************************
[LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
Training until validation scores don't improve for 200 rounds
[100]	valid_0's multi_logloss: 0.0498722
[200]	valid_0's multi_logloss: 0.038028
[300]	valid_0's multi_logloss: 0.0358066
[400]	valid_0's multi_logloss: 0.0361478
[500]	valid_0's multi_logloss: 0.0379597
Early stopping, best iteration is:
[340]	valid_0's multi_logloss: 0.0354344
预测的概率矩阵为：
[[9.99972032e-01 2.62406774e-05 1.17282152e-06 5.54230651e-07]
 [1.05242811e-05 6.50215805e-05 9.99924453e-01 6.93812546e-10]
 [1.93240868e-06 1.10384984e-07 3.76773426e-07 9.99997580e-01]
 ...
 [1.34894410e-02 3.84569683e-05 9.86471555e-01 5.46564350e-07]
 [9.99987431e-01 1.25532882e-05 1.03902298e-08 5.46727770e-09]
 [9.78722948e-01 1.06329839e-02 6.94192038e-03 3.70214810e-03]]
[607.0736049372186, 623.4313863731124, 508.02381607269535]
************************************ 4 ************************************
[LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
Training until validation scores don't improve for 200 rounds
[100]	valid_0's multi_logloss: 0.0564768
[200]	valid_0's multi_logloss: 0.0448698
[300]	valid_0's multi_logloss: 0.0446719
[400]	valid_0's multi_logloss: 0.0470399
Early stopping, best iteration is:
[250]	valid_0's multi_logloss: 0.0438853
预测的概率矩阵为：
[[9.99979692e-01 1.70821979e-05 1.27048476e-06 1.95571841e-06]
 [5.66207785e-05 4.02275314e-04 9.99541086e-01 1.82828519e-08]
 [2.62267451e-06 3.58613522e-07 4.78645006e-06 9.99992232e-01]
 ...
 [4.56636552e-02 5.69497433e-04 9.53758468e-01 8.37980573e-06]
 [9.99896785e-01 1.02796802e-04 2.46636563e-07 1.72061021e-07]
 [8.70911669e-01 1.73790185e-02 1.04478175e-01 7.23113697e-03]]
[607.0736049372186, 623.4313863731124, 508.02381607269535, 660.4867407547267]
************************************ 5 ************************************
[LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
Training until validation scores don't improve for 200 rounds
[100]	valid_0's multi_logloss: 0.0506398
[200]	valid_0's multi_logloss: 0.0396422
[300]	valid_0's multi_logloss: 0.0381065
[400]	valid_0's multi_logloss: 0.0390162
[500]	valid_0's multi_logloss: 0.0414986
Early stopping, best iteration is:
[324]	valid_0's multi_logloss: 0.0379497
预测的概率矩阵为：
[[9.99993352e-01 6.02902202e-06 1.13002685e-07 5.06277302e-07]
 [1.03959552e-05 5.03778956e-04 9.99485820e-01 5.07638601e-09]
 [1.92568065e-07 5.07155306e-08 4.94690856e-08 9.99999707e-01]
 ...
 [8.83103121e-03 2.51969353e-05 9.91142776e-01 9.96143937e-07]
 [9.99984791e-01 1.51997858e-05 5.62426491e-09 3.80450197e-09]
 [9.86084001e-01 8.75968498e-04 1.09742304e-02 2.06580027e-03]]
[607.0736049372186, 623.4313863731124, 508.02381607269535, 660.4867407547267, 539.2160054696063]
lgb_scotrainre_list: [607.0736049372186, 623.4313863731124, 508.02381607269535, 660.4867407547267, 539.2160054696063]
lgb_score_mean: 587.646310721472
lgb_score_std: 55.94453640571462
```

## 6.预测结果

```python
temp=pd.DataFrame(lgb_test)
result=pd.read_csv('sample_submit.csv')
result['label_0']=temp[0]
result['label_1']=temp[1]
result['label_2']=temp[2]
result['label_3']=temp[3]
result.to_csv('submit.csv',index=False)
```

