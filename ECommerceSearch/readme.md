## 比赛介绍

本次题目围绕电商领域搜索算法，开发者们可以通过基于阿里巴巴集团自研的高性能分布式搜索引擎问天引擎（提供高工程性能的电商智能搜索平台），可以快速迭代搜索算法，无需自主建设检索全链路环境。

 **本次评测的数据来自于淘宝搜索真实的业务场景，其中整个搜索商品集合按照商品的类别随机抽样保证了数据的多样性，搜索Query和相关的商品来自点击行为日志并通过模型+人工确认的方式完成校验保证了训练和测试数据的准确性。** 

比赛官网：[https://tianchi.aliyun.com/competition/entrance/531946/introduction](https://tianchi.aliyun.com/competition/entrance/531946/introduction)

## 学习内容

### 任务1：环境配置、实践数据下载

- 任务内容：
  - 从比赛官网下载数据集，并使用Python读取数据
  - 使用`jieba`对文本进行分词
  - 使用`TFIDF`对文本进行编码
  - 思考如何使用TFIDF计算文本相似度？
- 学习资料：[https://coggle.club/blog/tianchi-open-search](https://coggle.club/blog/tianchi-open-search)


### 任务2：词向量介绍与训练
