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
  - 写出使用TFIDF计算文本相似度的方法
- 学习资料：[https://coggle.club/blog/tianchi-open-search](https://coggle.club/blog/tianchi-open-search)


### 任务2：词向量介绍与训练

- 任务内容：
  - 使用任务1得到数据使用`gensim`训练词向量
  - 计算与`格力`相似的Top10单词
  - 使用词向量完成句子编码（例如单词编码为128维度，一个句子包含十个单词为10*128）
  - 对句子编码10*128进行求均值，转变为128维度
  - 扩展：你能使用计算得到的词向量，计算train.query.txt和corpus.tsv文本的相似度吗（train选择100条文本，corpus选择100条文本）？
- 学习资料：
  - [https://coggle.club/blog/tianchi-open-search](https://coggle.club/blog/tianchi-open-search)
  - [https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html)

### 任务3：IDF与词向量编码
- 任务内容：
  - 基于任务2的编码 & 训练集标注数据：筛选1k条train.query.txt文本，以及对应的在corpus.tsv文本的文本。
  - 使用任务2的编码方法对1k train 和 1k corpus的文本进行编码
  - 模拟文本检索的过程：train的文本128向量计算与corpus文本的向量相似度
  - 检索完成1k文本后，你能完成计算MRR吗（MRR计算为：1/正确corpus文本的次序）？
  - 扩展：你能使用单词的IDF筛选单词（IDF可以从任务1得到），然后再对句子进行编码吗？
- 学习资料：
  - [https://coggle.club/blog/tianchi-open-search](https://coggle.club/blog/tianchi-open-search)


### 任务4：文本编码与提交
- 任务内容：
  - 使用任务2 & 任务3的思路对dev.query.txt和corpus.tsv进行编码
  - 将编码结果编码为比赛需要的格式
  - 将结果打包提交到天池，得到具体得到
- 学习资料：
  - [https://coggle.club/blog/tianchi-open-search](https://coggle.club/blog/tianchi-open-search)
