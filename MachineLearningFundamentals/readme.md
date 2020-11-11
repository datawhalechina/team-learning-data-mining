# 机器学习算法基础（上）

## 基本信息

- 学习周期：11天，每天平均花费时间3小时-5小时不等，根据个人学习接受能力强弱有所浮动。
- 学习形式：理论学习 + 练习
- 人群定位：有概率论、矩阵运算、求导、泰勒展开等基础数学知识。
- 先修内容：[Python编程语言](https://github.com/datawhalechina/team-learning-program/tree/master/Python-Language)，[概率统计](https://github.com/datawhalechina/team-learning-data-mining/tree/master/ProbabilityStatistics)
- 难度系数：中

## 任务安排




### Task01：机器学习概述（1天）



<b>理论部分</b>

- 机器学习介绍：机器学习是什么，怎么来的，理论基础是什么，为了解决什么问题。
- 机器学习分类：
    - 按学习方式分：有监督、无监督、半监督 
    - 按任务类型分：回归、分类、聚类、降维 生成模型与判别模型
- 机器学习方法三要素：
    - **模型** 
    - **策略**：损失函数 
    - **算法**：梯度下降法、牛顿法、拟牛顿法
    - 模型评估指标：R2、RMSE、accuracy、precision、recall、F1、ROC、AUC、Confusion Matrix 
    - 复杂度度量：偏差与方差、过拟合与欠拟合、结构风险与经验风险、泛化能力、正则化 
    - 模型选择：正则化、交叉验证 
    - 采样：样本不均衡 
    - 特征处理：归一化、标准化、离散化、one-hot编码 
    - 模型调优：网格搜索寻优、随机搜索寻优



### Task02：线性回归（2天）

<b>理论部分</b>

- 模型建立：线性回归原理、线性回归模型
- 学习策略：线性回归损失函数、代价函数、目标函数
- 算法求解：梯度下降法、牛顿法、拟牛顿法等
- 线性回归的评估指标
- sklearn参数详解


<b>练习部分</b>


- 基于线性回归的房价预测问题
- 利用`sklearn`解决回归问题
- `sklearn.linear_model.LinearRegression`


### Task03：逻辑回归（2天）

<b>理论部分</b>

- 逻辑回归与线性回归的联系与区别
- 模型建立：逻辑回归原理、逻辑回归模型
- 学习策略：逻辑回归损失函数、推导及优化
- 算法求解：批量梯度下降
- 正则化与模型评估指标
- 逻辑回归的优缺点
- 样本不均衡问题
- sklearn参数详解


<b>练习部分</b>


- 利用`sklearn`解决分类问题
- `sklearn.linear_model.LogisticRegression`
- 利用梯度下降法将相同的数据分类，画图和sklearn的结果相比较
- 利用牛顿法实现结果，画图和sklearn的结果相比较，并比较牛顿法和梯度下降法迭代收敛的次数


### Task04：决策树（2天）

<b>理论部分</b>

- 特征选择：信息增益（熵、联合熵、条件熵）、信息增益比、基尼系数
- 决策树生成：ID3决策树、C4.5决策树、CART决策树（CART分类树、CART回归树）
- 决策树剪枝
- sklearn参数详解

<b>练习部分</b>


- 利用`sklearn`解决分类问题和回归预测。
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.tree.DecisionTreeRegressor`


### Task05：聚类（2天）
<b>理论部分</b>

- 相关概念
    - 无监督学习
    - 聚类的定义
- 常用距离公式
    - 曼哈顿距离
    - 欧式距离
    - 闵可夫斯基距离
    - 切比雪夫距离
    - 夹角余弦
    - 汉明距离
    - 杰卡德相似系数
    - 杰卡德距离
- K-Means聚类：聚类过程和原理、算法流程、算法优化（k-means++、Mini Batch K-Means）
- 层次聚类：Agglomerative Clustering过程和原理
- 密度聚类：DBSCAN过程和原理
- 谱聚类：谱聚类原理（邻接矩阵、度矩阵、拉普拉斯矩阵、RatioCut、Ncut）和过程
- 高斯混合聚类：GMM过程和原理、EM算法原理、利用EM算法估计高斯混合聚类参数
- sklearn参数详解

<b>练习部分</b>


- 利用`sklearn`解决聚类问题。
- `sklearn.cluster.KMeans`


### Task06：朴素贝叶斯（2天）
<b>理论部分</b>
- 相关概念
    - 生成模型
    - 判别模型
- 朴素贝叶斯基本原理
    - 条件概率公式
    - 乘法公式
    - 全概率公式
    - 贝叶斯定理
    - 特征条件独立假设
    - 后验概率最大化
    - 拉普拉斯平滑
- 朴素贝叶斯的三种形式
    - 高斯型
    - 多项式型
    - 伯努利型
- 极值问题情况下的每个类的分类概率
- 下溢问题如何解决
- 零概率问题如何解决
- sklearn参数详解

<b>练习部分</b>


- 利用`sklearn`解决聚类问题。
- `sklearn.naive_bayes.GaussianNB`


---
# 机器学习算法基础（下）

## 基本信息

- 学习周期：10天，每天平均花费时间2小时-5小时不等，根据个人学习接受能力强弱有所浮动。
- 学习形式：理论学习 + 练习
- 人群定位：有概率论、矩阵运算、微积分、最优化理论等基础数学知识。
- 先修内容：[Python编程语言](https://github.com/datawhalechina/team-learning-program/tree/master/Python-Language)，[概率统计](https://github.com/datawhalechina/team-learning-data-mining/tree/master/ProbabilityStatistics)
- 难度系数：中




## 任务安排

### Task01：线性回归（2天）

<b>理论部分</b>
- 模型建立：线性回归原理、线性回归模型
- 学习策略：线性回归损失函数、代价函数、目标函数
- 算法求解：梯度下降法、牛顿法、拟牛顿法等
- 线性回归的评估指标
- sklearn参数详解


<b>练习部分</b>

- 基于线性回归的房价预测问题
- 利用`sklearn`解决回归问题
- `sklearn.linear_model.LinearRegression`


### Task02：朴素贝叶斯（2天）
<b>理论部分</b>
- 相关概念
    - 生成模型
    - 判别模型
- 朴素贝叶斯基本原理
    - 条件概率公式
    - 乘法公式
    - 全概率公式
    - 贝叶斯定理
    - 特征条件独立假设
    - 后验概率最大化
    - 拉普拉斯平滑
- 朴素贝叶斯的三种形式
    - 高斯型
    - 多项式型
    - 伯努利型
- 极值问题情况下的每个类的分类概率
- 下溢问题如何解决
- 零概率问题如何解决
- sklearn参数详解

<b>练习部分</b>

- 利用`sklearn`解决聚类问题。
- `sklearn.naive_bayes.GaussianNB`




### Task03：EM算法（2天）
<b>理论部分</b>
- 相关概念
    - 极大似然估计法
    - 贝叶斯估计方法
- EM基本原理
    - E步
    - M步
    - 推导、证明
    - 高斯混合分布


<b>练习部分</b>

- 算法实现



### Task04：条件随机场（2天）
<b>理论部分</b>
- 前提：相关概念
    - 马尔可夫过程
    - 隐马尔科夫算法
    - 
- 条件随机场
    - 转移特征和状态特征
    - 矩阵形式
- 条件随机场三问题
    - 计算问题
    - 学习问题
    - 预测问题


<b>练习部分</b>

- 利用高维特比算法计算给定输入序列对应的最优输出序列


### Task05：SVM（2天）
<b>理论部分</b>
- 概念：最大超平面
- 数学知识：拉格朗日乘子
- SVM 硬间隔优化公式
- SVM 软间隔原理
- 核函数

选修 ： SMO 求解SVM

<b>练习部分</b>

- 算法实现



---
# 贡献人员


姓名 | 博客|备注
---|---|---
肖然||中国科学院硕士
谢文昕||上海交通大学博士
高立业||太原理工大学硕士
赵楠||福州大学硕士
杨开漠 | [Github](https://github.com/km1994)|五邑大学计算机硕士
张雨||复旦大学博士
马燕鹏|[CSDN](https://lsgogroup.blog.csdn.net/)<br>微信公众号：LSGO软件技术团队|华北电力大学
张峰|[Github](https://github.com/Hirotransfer)|安徽工业大学硕士
