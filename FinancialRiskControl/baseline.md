# baseline

```python
import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
```


```python
train = pd.read_csv('train.csv')
testA = pd.read_csv('testA.csv')
```


```python
train.head()
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
      <th>3</th>
      <td>3</td>
      <td>11000.0</td>
      <td>3</td>
      <td>7.26</td>
      <td>340.96</td>
      <td>A</td>
      <td>A4</td>
      <td>46854.0</td>
      <td>10+ years</td>
      <td>1</td>
      <td>...</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3000.0</td>
      <td>3</td>
      <td>12.99</td>
      <td>101.07</td>
      <td>C</td>
      <td>C2</td>
      <td>54.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>




```python
data = pd.concat([train, testA], axis=0, ignore_index=True)
```

## 数据预处理
- 可以看到很多变量不能直接训练，比如grade、subGrade、employmentLength、issueDate、earliesCreditLine，需要进行预处理


```python
print(sorted(data['grade'].unique()))
print(sorted(data['subGrade'].unique()))
```

    ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
    


```python
data['employmentLength'].value_counts(dropna=False).sort_index()
```




    1 year        65671
    10+ years    328525
    2 years       90565
    3 years       80163
    4 years       59818
    5 years       62645
    6 years       46582
    7 years       44230
    8 years       45168
    9 years       37866
    < 1 year      80226
    NaN           58541
    Name: employmentLength, dtype: int64



- 首先对employmentLength进行转换到数值


```python
data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
data['employmentLength'].replace('< 1 year', '0 years', inplace=True)

def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
    
data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)
```


```python
data['employmentLength'].value_counts(dropna=False).sort_index()
```




    0.0      80226
    1.0      65671
    2.0      90565
    3.0      80163
    4.0      59818
    5.0      62645
    6.0      46582
    7.0      44230
    8.0      45168
    9.0      37866
    10.0    328525
    NaN      58541
    Name: employmentLength, dtype: int64



- 对earliesCreditLine进行预处理


```python
data['earliesCreditLine'].sample(5)
```




    375743    Jun-2003
    361340    Jul-1999
    716602    Aug-1995
    893559    Oct-1982
    221525    Nov-2004
    Name: earliesCreditLine, dtype: object




```python
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))
```


```python
data['earliesCreditLine'].describe()
```




    count    1000000.000000
    mean        1998.688632
    std            7.606231
    min         1944.000000
    25%         1995.000000
    50%         2000.000000
    75%         2004.000000
    max         2015.000000
    Name: earliesCreditLine, dtype: float64




```python
data.head()
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
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
      <th>n2.2</th>
      <th>n2.3</th>
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
      <td>2.0</td>
      <td>2</td>
      <td>...</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>5.0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>8.0</td>
      <td>0</td>
      <td>...</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>11000.0</td>
      <td>3</td>
      <td>7.26</td>
      <td>340.96</td>
      <td>A</td>
      <td>A4</td>
      <td>46854.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>...</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3000.0</td>
      <td>3</td>
      <td>12.99</td>
      <td>101.07</td>
      <td>C</td>
      <td>C2</td>
      <td>54.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



- 类别特征处理


```python
# 部分类别特征
cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                 'applicationType', 'initialListStatus', 'title', 'policyCode']
for f in cate_features:
    print(f, '类型数：', data[f].nunique())
```

    grade 类型数： 7
    subGrade 类型数： 35
    employmentTitle 类型数： 298101
    homeOwnership 类型数： 6
    verificationStatus 类型数： 3
    purpose 类型数： 14
    postCode 类型数： 935
    regionCode 类型数： 51
    applicationType 类型数： 2
    initialListStatus 类型数： 2
    title 类型数： 47903
    policyCode 类型数： 1
    


```python
# 类型数在2之上，又不是高维稀疏的
data = pd.get_dummies(data, columns=['grade', 'subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'], drop_first=True)
```


```python
# 高维类别特征需要进行转换
for f in ['employmentTitle', 'postCode', 'title']:
    data[f+'_cnts'] = data.groupby([f])['id'].transform('count')
    data[f+'_rank'] = data.groupby([f])['id'].rank(ascending=False).astype(int)
    del data[f]
```

## 训练数据/测试数据准备


```python
features = [f for f in data.columns if f not in ['id','issueDate','isDefault']]

train = data[data.isDefault.notnull()].reset_index(drop=True)
test = data[data.isDefault.isnull()].reset_index(drop=True)

x_train = train[features]
x_test = test[features]

y_train = train['isDefault']
```

## 模型训练
- 直接构建了一个函数，可以调用三种树模型，方便快捷


```python
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            test_matrix = clf.DMatrix(test_x)
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)
                 
        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test
```


```python
def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test

def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat") 
    return cat_train, cat_test
```


```python
lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
```

    ************************************ 1 ************************************
    Training until validation scores don't improve for 200 rounds.
    [200]	training's auc: 0.742884	valid_1's auc: 0.73055
    [400]	training's auc: 0.755686	valid_1's auc: 0.731888
    [600]	training's auc: 0.766421	valid_1's auc: 0.731988
    [800]	training's auc: 0.776244	valid_1's auc: 0.731868
    Early stopping, best iteration is:
    [656]	training's auc: 0.769146	valid_1's auc: 0.732081
    [0.7320814878889421]
    ************************************ 2 ************************************
    Training until validation scores don't improve for 200 rounds.
    [200]	training's auc: 0.74372	valid_1's auc: 0.726466
    [400]	training's auc: 0.756459	valid_1's auc: 0.727727
    [600]	training's auc: 0.767156	valid_1's auc: 0.727776
    Early stopping, best iteration is:
    [520]	training's auc: 0.762985	valid_1's auc: 0.727902
    [0.7320814878889421, 0.7279015876934286]
    ************************************ 3 ************************************
    Training until validation scores don't improve for 200 rounds.
    [200]	training's auc: 0.742884	valid_1's auc: 0.731466
    [400]	training's auc: 0.755466	valid_1's auc: 0.732748
    [600]	training's auc: 0.766313	valid_1's auc: 0.733069
    [800]	training's auc: 0.776349	valid_1's auc: 0.732892
    Early stopping, best iteration is:
    [694]	training's auc: 0.771133	valid_1's auc: 0.73312
    [0.7320814878889421, 0.7279015876934286, 0.7331203287449972]
    ************************************ 4 ************************************
    Training until validation scores don't improve for 200 rounds.
    [200]	training's auc: 0.742632	valid_1's auc: 0.730114
    [400]	training's auc: 0.755357	valid_1's auc: 0.731443
    [600]	training's auc: 0.765983	valid_1's auc: 0.731566
    [800]	training's auc: 0.776112	valid_1's auc: 0.731805
    Early stopping, best iteration is:
    [706]	training's auc: 0.771324	valid_1's auc: 0.731887
    [0.7320814878889421, 0.7279015876934286, 0.7331203287449972, 0.731886588682118]
    ************************************ 5 ************************************
    Training until validation scores don't improve for 200 rounds.
    [200]	training's auc: 0.743113	valid_1's auc: 0.729226
    [400]	training's auc: 0.7559	valid_1's auc: 0.730816
    [600]	training's auc: 0.766388	valid_1's auc: 0.73092
    [800]	training's auc: 0.77627	valid_1's auc: 0.731029
    [1000]	training's auc: 0.785791	valid_1's auc: 0.730933
    Early stopping, best iteration is:
    [883]	training's auc: 0.780369	valid_1's auc: 0.731096
    [0.7320814878889421, 0.7279015876934286, 0.7331203287449972, 0.731886588682118, 0.7310960057774112]
    lgb_scotrainre_list: [0.7320814878889421, 0.7279015876934286, 0.7331203287449972, 0.731886588682118, 0.7310960057774112]
    lgb_score_mean: 0.7312171997573793
    lgb_score_std: 0.001779041696522632
    


```python
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
```

    ************************************ 1 ************************************
    [0]	train-auc:0.677293	eval-auc:0.678869
    Multiple eval metrics have been passed: 'eval-auc' will be used for early stopping.
    
    Will train until eval-auc hasn't improved in 200 rounds.
    [200]	train-auc:0.727527	eval-auc:0.723771
    [400]	train-auc:0.73516	eval-auc:0.727725
    [600]	train-auc:0.740458	eval-auc:0.729631
    [800]	train-auc:0.744963	eval-auc:0.730829
    [1000]	train-auc:0.748802	eval-auc:0.731495
    [1200]	train-auc:0.752295	eval-auc:0.732074
    [1400]	train-auc:0.755574	eval-auc:0.732421
    [1600]	train-auc:0.758671	eval-auc:0.732674
    [1800]	train-auc:0.761605	eval-auc:0.732964
    [2000]	train-auc:0.764627	eval-auc:0.733111
    [2200]	train-auc:0.767443	eval-auc:0.733201
    [2400]	train-auc:0.770204	eval-auc:0.733224
    Stopping. Best iteration:
    [2328]	train-auc:0.7692	eval-auc:0.733246
    
    [0.7332460852050292]
    ************************************ 2 ************************************
    [0]	train-auc:0.677718	eval-auc:0.672523
    Multiple eval metrics have been passed: 'eval-auc' will be used for early stopping.
    
    Will train until eval-auc hasn't improved in 200 rounds.
    [200]	train-auc:0.728628	eval-auc:0.720255
    [400]	train-auc:0.736149	eval-auc:0.724308
    [600]	train-auc:0.741354	eval-auc:0.726443
    [800]	train-auc:0.745611	eval-auc:0.72746
    [1000]	train-auc:0.749627	eval-auc:0.728194
    [1200]	train-auc:0.753176	eval-auc:0.728711
    [1400]	train-auc:0.756476	eval-auc:0.72899
    [1600]	train-auc:0.759574	eval-auc:0.729224
    [1800]	train-auc:0.762608	eval-auc:0.729501
    [2000]	train-auc:0.765549	eval-auc:0.729627
    [2200]	train-auc:0.768304	eval-auc:0.729782
    [2400]	train-auc:0.771131	eval-auc:0.729922
    [2600]	train-auc:0.773769	eval-auc:0.729961
    [2800]	train-auc:0.776371	eval-auc:0.72999
    Stopping. Best iteration:
    [2697]	train-auc:0.775119	eval-auc:0.730036
    
    [0.7332460852050292, 0.7300358478747684]
    ************************************ 3 ************************************
    [0]	train-auc:0.676641	eval-auc:0.67765
    Multiple eval metrics have been passed: 'eval-auc' will be used for early stopping.
    
    Will train until eval-auc hasn't improved in 200 rounds.
    [200]	train-auc:0.72757	eval-auc:0.724632
    [400]	train-auc:0.735185	eval-auc:0.728571
    [600]	train-auc:0.740671	eval-auc:0.73067
    [800]	train-auc:0.745049	eval-auc:0.731899
    [1000]	train-auc:0.748976	eval-auc:0.732787
    [1200]	train-auc:0.752383	eval-auc:0.73321
    [1400]	train-auc:0.75564	eval-auc:0.733548
    [1600]	train-auc:0.758796	eval-auc:0.733825
    [1800]	train-auc:0.761717	eval-auc:0.734007
    [2000]	train-auc:0.76459	eval-auc:0.734193
    [2200]	train-auc:0.767399	eval-auc:0.734261
    [2400]	train-auc:0.770174	eval-auc:0.734362
    [2600]	train-auc:0.772818	eval-auc:0.734369
    [2800]	train-auc:0.775568	eval-auc:0.734391
    [3000]	train-auc:0.777985	eval-auc:0.73444
    [3200]	train-auc:0.780514	eval-auc:0.734477
    [3400]	train-auc:0.782893	eval-auc:0.734427
    Stopping. Best iteration:
    [3207]	train-auc:0.780621	eval-auc:0.734494
    
    [0.7332460852050292, 0.7300358478747684, 0.7344942212088965]
    ************************************ 4 ************************************
    [0]	train-auc:0.677768	eval-auc:0.677179
    Multiple eval metrics have been passed: 'eval-auc' will be used for early stopping.
    
    Will train until eval-auc hasn't improved in 200 rounds.
    [200]	train-auc:0.727614	eval-auc:0.72295
    [400]	train-auc:0.735165	eval-auc:0.726994
    [600]	train-auc:0.740498	eval-auc:0.729116
    [800]	train-auc:0.744884	eval-auc:0.730417
    [1000]	train-auc:0.748782	eval-auc:0.731318
    [1200]	train-auc:0.75225	eval-auc:0.731899
    [1400]	train-auc:0.755505	eval-auc:0.732295
    [1600]	train-auc:0.758618	eval-auc:0.732629
    [1800]	train-auc:0.76176	eval-auc:0.733046
    [2000]	train-auc:0.764736	eval-auc:0.733189
    [2200]	train-auc:0.767476	eval-auc:0.733276
    [2400]	train-auc:0.770154	eval-auc:0.733409
    [2600]	train-auc:0.772874	eval-auc:0.733469
    [2800]	train-auc:0.77541	eval-auc:0.733405
    Stopping. Best iteration:
    [2644]	train-auc:0.773429	eval-auc:0.733488
    
    [0.7332460852050292, 0.7300358478747684, 0.7344942212088965, 0.7334876284761012]
    ************************************ 5 ************************************
    [0]	train-auc:0.677768	eval-auc:0.676353
    Multiple eval metrics have been passed: 'eval-auc' will be used for early stopping.
    
    Will train until eval-auc hasn't improved in 200 rounds.
    [200]	train-auc:0.728072	eval-auc:0.722913
    [400]	train-auc:0.735517	eval-auc:0.726582
    [600]	train-auc:0.740782	eval-auc:0.728449
    [800]	train-auc:0.745258	eval-auc:0.729653
    [1000]	train-auc:0.749185	eval-auc:0.730489
    [1200]	train-auc:0.752723	eval-auc:0.731038
    [1400]	train-auc:0.755985	eval-auc:0.731466
    [1600]	train-auc:0.759166	eval-auc:0.731758
    [1800]	train-auc:0.762205	eval-auc:0.73199
    [2000]	train-auc:0.765197	eval-auc:0.732145
    [2200]	train-auc:0.767976	eval-auc:0.732194
    Stopping. Best iteration:
    [2191]	train-auc:0.767852	eval-auc:0.732213
    
    [0.7332460852050292, 0.7300358478747684, 0.7344942212088965, 0.7334876284761012, 0.7322134048106561]
    xgb_scotrainre_list: [0.7332460852050292, 0.7300358478747684, 0.7344942212088965, 0.7334876284761012, 0.7322134048106561]
    xgb_score_mean: 0.7326954375150903
    xgb_score_std: 0.0015147392354657807
    


```python
cat_train, cat_test = cat_model(x_train, y_train, x_test)
```

    ************************************ 1 ************************************
    0:	learn: 0.4415198	test: 0.4387088	best: 0.4387088 (0)	total: 111ms	remaining: 37m 6s
    500:	learn: 0.3772118	test: 0.3759665	best: 0.3759665 (500)	total: 37.7s	remaining: 24m 25s
    1000:	learn: 0.3756709	test: 0.3752058	best: 0.3752058 (1000)	total: 1m 14s	remaining: 23m 41s
    1500:	learn: 0.3745785	test: 0.3748423	best: 0.3748423 (1500)	total: 1m 52s	remaining: 23m 7s
    2000:	learn: 0.3736834	test: 0.3746564	best: 0.3746564 (2000)	total: 2m 29s	remaining: 22m 28s
    2500:	learn: 0.3728568	test: 0.3745180	best: 0.3745165 (2492)	total: 3m 7s	remaining: 21m 52s
    3000:	learn: 0.3720793	test: 0.3744201	best: 0.3744198 (2998)	total: 3m 44s	remaining: 21m 14s
    Stopped by overfitting detector  (50 iterations wait)
    
    bestTest = 0.3744006318
    bestIteration = 3086
    
    Shrink model to first 3087 iterations.
    [0.7326058985428212]
    ************************************ 2 ************************************
    0:	learn: 0.4406928	test: 0.4420714	best: 0.4420714 (0)	total: 53.3ms	remaining: 17m 46s
    500:	learn: 0.3765250	test: 0.3787287	best: 0.3787287 (500)	total: 38.7s	remaining: 25m 8s
    1000:	learn: 0.3749822	test: 0.3779503	best: 0.3779503 (998)	total: 1m 16s	remaining: 24m 18s
    1500:	learn: 0.3738772	test: 0.3775654	best: 0.3775654 (1500)	total: 1m 54s	remaining: 23m 34s
    2000:	learn: 0.3729354	test: 0.3773407	best: 0.3773401 (1999)	total: 2m 33s	remaining: 22m 56s
    2500:	learn: 0.3721077	test: 0.3771987	best: 0.3771971 (2496)	total: 3m 10s	remaining: 22m 15s
    3000:	learn: 0.3713621	test: 0.3771114	best: 0.3771114 (3000)	total: 3m 49s	remaining: 21m 37s
    Stopped by overfitting detector  (50 iterations wait)
    
    bestTest = 0.3770400469
    bestIteration = 3382
    
    Shrink model to first 3383 iterations.
    [0.7326058985428212, 0.7292909146788396]
    ************************************ 3 ************************************
    0:	learn: 0.4408230	test: 0.4418939	best: 0.4418939 (0)	total: 59.1ms	remaining: 19m 42s
    500:	learn: 0.3767851	test: 0.3776319	best: 0.3776319 (500)	total: 40.4s	remaining: 26m 12s
    1000:	learn: 0.3752331	test: 0.3768292	best: 0.3768292 (1000)	total: 1m 20s	remaining: 25m 19s
    1500:	learn: 0.3741550	test: 0.3764926	best: 0.3764926 (1500)	total: 2m	remaining: 24m 39s
    2000:	learn: 0.3732520	test: 0.3762840	best: 0.3762832 (1992)	total: 2m 40s	remaining: 24m 2s
    2500:	learn: 0.3724303	test: 0.3761303	best: 0.3761279 (2490)	total: 3m 20s	remaining: 23m 22s
    3000:	learn: 0.3716684	test: 0.3760402	best: 0.3760395 (2995)	total: 4m	remaining: 22m 42s
    3500:	learn: 0.3709308	test: 0.3759509	best: 0.3759502 (3495)	total: 4m 40s	remaining: 22m 2s
    4000:	learn: 0.3702269	test: 0.3759039	best: 0.3759027 (3993)	total: 5m 20s	remaining: 21m 20s
    4500:	learn: 0.3695477	test: 0.3758698	best: 0.3758663 (4459)	total: 6m	remaining: 20m 40s
    Stopped by overfitting detector  (50 iterations wait)
    
    bestTest = 0.3758663409
    bestIteration = 4459
    
    Shrink model to first 4460 iterations.
    [0.7326058985428212, 0.7292909146788396, 0.7341207611812285]
    ************************************ 4 ************************************
    0:	learn: 0.4408778	test: 0.4413264	best: 0.4413264 (0)	total: 46.6ms	remaining: 15m 32s
    500:	learn: 0.3768022	test: 0.3777678	best: 0.3777678 (500)	total: 40.3s	remaining: 26m 7s
    1000:	learn: 0.3753097	test: 0.3769403	best: 0.3769403 (1000)	total: 1m 20s	remaining: 25m 24s
    1500:	learn: 0.3742418	test: 0.3765698	best: 0.3765698 (1500)	total: 2m	remaining: 24m 41s
    2000:	learn: 0.3733478	test: 0.3763500	best: 0.3763496 (1998)	total: 2m 40s	remaining: 23m 59s
    2500:	learn: 0.3725263	test: 0.3762101	best: 0.3762093 (2488)	total: 3m 20s	remaining: 23m 19s
    3000:	learn: 0.3717486	test: 0.3760966	best: 0.3760966 (2999)	total: 3m 59s	remaining: 22m 36s
    Stopped by overfitting detector  (50 iterations wait)
    
    bestTest = 0.3760182133
    bestIteration = 3432
    
    Shrink model to first 3433 iterations.
    [0.7326058985428212, 0.7292909146788396, 0.7341207611812285, 0.7324483603137153]
    ************************************ 5 ************************************
    0:	learn: 0.4409876	test: 0.4409159	best: 0.4409159 (0)	total: 52.3ms	remaining: 17m 26s
    500:	learn: 0.3768055	test: 0.3776229	best: 0.3776229 (500)	total: 38s	remaining: 24m 38s
    1000:	learn: 0.3752600	test: 0.3768397	best: 0.3768397 (1000)	total: 1m 15s	remaining: 23m 57s
    1500:	learn: 0.3741843	test: 0.3764855	best: 0.3764855 (1500)	total: 1m 53s	remaining: 23m 16s
    2000:	learn: 0.3732691	test: 0.3762491	best: 0.3762490 (1998)	total: 2m 31s	remaining: 22m 40s
    2500:	learn: 0.3724407	test: 0.3761154	best: 0.3761154 (2500)	total: 3m 9s	remaining: 22m 5s
    3000:	learn: 0.3716764	test: 0.3760184	best: 0.3760184 (3000)	total: 3m 47s	remaining: 21m 26s
    3500:	learn: 0.3709545	test: 0.3759453	best: 0.3759453 (3500)	total: 4m 24s	remaining: 20m 47s
    Stopped by overfitting detector  (50 iterations wait)
    
    bestTest = 0.3759421091
    bestIteration = 3544
    
    Shrink model to first 3545 iterations.
    [0.7326058985428212, 0.7292909146788396, 0.7341207611812285, 0.7324483603137153, 0.7312334660628076]
    cat_scotrainre_list: [0.7326058985428212, 0.7292909146788396, 0.7341207611812285, 0.7324483603137153, 0.7312334660628076]
    cat_score_mean: 0.7319398801558824
    cat_score_std: 0.001610863965629903
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-25-2e9bafef31e8> in <module>
    ----> 1 cat_train, cat_test = cat_model(x_train, y_train, x_test)
    

    TypeError: 'NoneType' object is not iterable



```python
rh_test = lgb_test*0.5 + xgb_test*0.5
```


```python
testA['isDefault'] = rh_test
```


```python
testA[['id','isDefault']].to_csv('test_sub.csv', index=False)
```


```python

```
