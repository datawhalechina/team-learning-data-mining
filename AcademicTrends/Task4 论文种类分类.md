# 任务4：论文种类分类

## 4.1 任务说明

- 学习主题：论文分类（数据建模任务），利用已有数据建模，对新论文进行类别分类；
- 学习内容：使用论文标题完成类别分类；
- 学习成果：学会文本分类的基本方法、`TF-IDF`等；

## 4.2 数据处理步骤

在原始arxiv论文中论文都有对应的类别，而论文类别是作者填写的。在本次任务中我们可以借助论文的标题和摘要完成：

- 对论文标题和摘要进行处理；
- 对论文类别进行处理；
- 构建文本分类模型；

## 4.3 文本分类思路

- 思路1：TF-IDF+机器学习分类器

直接使用TF-IDF对文本提取特征，使用分类器进行分类，分类器的选择上可以使用SVM、LR、XGboost等

- 思路2：FastText

FastText是入门款的词向量，利用Facebook提供的FastText工具，可以快速构建分类器

- 思路3：WordVec+深度学习分类器

WordVec是进阶款的词向量，并通过构建深度学习分类完成分类。深度学习分类的网络结构可以选择TextCNN、TextRnn或者BiLSTM。

- 思路4：Bert词向量

Bert是高配款的词向量，具有强大的建模学习能力。

## 4.4 具体代码实现以及讲解

为了方便大家入门文本分类，我们选择思路1和思路2给大家讲解。首先完成字段读取：

```python
data  = [] #初始化
#使用with语句优势：1.自动关闭文件句柄；2.自动显示（处理）文件读取数据异常
with open("arxiv-metadata-oai-snapshot.json", 'r') as f: 
    for idx, line in enumerate(f): 
        d = json.loads(line)
        d = {'title': d['title'], 'categories': d['categories'], 'abstract': d['abstract']}
        data.append(d)
        
        # 选择部分数据
        if idx > 200000:
            break
        
data = pd.DataFrame(data) #将list变为dataframe格式，方便使用pandas进行分析
```

为了方便数据的处理，我们可以将标题和摘要拼接一起完成分类。

```python
data['text'] = data['title'] + data['abstract']

data['text'] = data['text'].apply(lambda x: x.replace('\n',' '))
data['text'] = data['text'].apply(lambda x: x.lower())
data = data.drop(['abstract', 'title'], axis=1)
```

由于原始论文有可能有多个类别，所以也需要处理：

```python
# 多个类别，包含子分类
data['categories'] = data['categories'].apply(lambda x : x.split(' '))

# 单个类别，不包含子分类
data['categories_big'] = data['categories'].apply(lambda x : [xx.split('.')[0] for xx in x])
```

然后将类别进行编码，这里类别是多个，所以需要多编码：

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
data_label = mlb.fit_transform(data['categories_big'].iloc[:])
```

### 4.4.1 思路1

思路1使用TFIDF提取特征，限制最多4000个单词：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=4000)
data_tfidf = vectorizer.fit_transform(data['text'].iloc[:])
```

由于这里是多标签分类，可以使用sklearn的多标签分类进行封装：

```python
# 划分训练集和验证集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_tfidf, data_label,
                                                 test_size = 0.2,random_state = 1)

# 构建多标签分类模型
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
clf = MultiOutputClassifier(MultinomialNB()).fit(x_train, y_train)
```

验证模型的精度：

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, clf.predict(x_test)))
```

### 4.4.2 思路2

思路2使用深度学习模型，单词进行词嵌入然后训练。首先按照文本划分数据集：

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['text'].iloc[:], data_label,
                                                 test_size = 0.2,random_state = 1)
```

将数据集处理进行编码，并进行截断：

```python
# parameter
max_features= 500
max_len= 150
embed_size=100
batch_size = 128
epochs = 5

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

tokens = Tokenizer(num_words = max_features)
tokens.fit_on_texts(list(x_train)+list(x_test))

x_sub_train = tokens.texts_to_sequences(x_train)
x_sub_test = tokens.texts_to_sequences(x_test)

x_sub_train=sequence.pad_sequences(x_sub_train, maxlen=max_len)
x_sub_test=sequence.pad_sequences(x_sub_test, maxlen=max_len)
```

定义模型并完成训练：

```python
# LSTM model
# Keras Layers:
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D# Keras Callback Functions:
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.models import Model
from keras.optimizers import Adam

sequence_input = Input(shape=(max_len, ))
x = Embedding(max_features, embed_size,trainable = False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
preds = Dense(20, activation="sigmoid")(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])
model.fit(x_sub_train, y_train, batch_size=batch_size, epochs=epochs)
```

