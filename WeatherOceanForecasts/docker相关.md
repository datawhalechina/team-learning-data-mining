“AI Earth”人工智能创新挑战赛Docker提交

  

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

01

简介

  
本次竞赛的Docker提交大致可以分为两小块：

1.  线下文件准备好：包括**DockerFile，代码，预测的代码**；
2.  Build,pull,提交,等待好消息;

如果之前没有提交过docker，可以根据这篇教程熟悉一下：https://tianchi.aliyun.com/forum/postDetail\?spm=5176.12586969.1002.9.51df4127FoZKeL\&postId=165595  

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

02

文件准备

### 1\. requirement.txt

 -    运行代码所依赖的python库，缺什么就把需要装的文件放在requirement下面

```
numpytensorflow==2.2.0 
```

### 2\. 运行的代码

#### 放在code下面即可

```
import tensorflow as tfimport tensorflow.keras.backend as Kfrom tensorflow.keras.layers import *from tensorflow.keras.models import *from tensorflow.keras.optimizers import *from tensorflow.keras.callbacks import *from tensorflow.keras.layers import Input import numpy as npimport osimport zipfiledef RMSE(y_true, y_pred):    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))def build_model():      inp    = Input(shape=(12,24,72,4))          x_4    = Dense(1, activation='relu')(inp)       x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))         x = Dense(64, activation='relu')(x_1)      x = Dropout(0.25)(x)     x = Dense(32, activation='relu')(x)       x = Dropout(0.25)(x)      output = Dense(24, activation='linear')(x)       model  = Model(inputs=inp, outputs=output)    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99)     model.compile(optimizer=adam, loss=RMSE)    return model model = build_model()model.load_weights('./user_data/model_data/model_mlp_baseline.h5')test_path = './tcdata/enso_round1_test_20210201/'### 1. 测试数据读取files = os.listdir(test_path)test_feas_dict = {}for file in files:    test_feas_dict[file] = np.load(test_path + file)    ### 2. 结果预测test_predicts_dict = {}for file_name,val in test_feas_dict.items():    test_predicts_dict[file_name] = model.predict(val).reshape(-1,)#     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])### 3.存储预测结果for file_name,val in test_predicts_dict.items():     np.save('./result/' + file_name,val)#打包目录为zip文件（未压缩）def make_zip(source_dir='./result/', output_filename = 'result.zip'):    zipf = zipfile.ZipFile(output_filename, 'w')    pre_len = len(os.path.dirname(source_dir))    source_dirs = os.walk(source_dir)    print(source_dirs)    for parent, dirnames, filenames in source_dirs:        print(parent, dirnames)        for filename in filenames:            if '.npy' not in filename:                continue            pathfile = os.path.join(parent, filename)            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径            zipf.write(pathfile, arcname)    zipf.close()make_zip() 
```

#### 3\. run.sh

 -    运行预测的代码

```
#!/bin/shCURDIR="`dirname $0`" #获取此脚本所在目录echo $CURDIRcd $CURDIR #切换到该脚本所在目录python /code/mlp_predict.py
```

### 4\. DockerFile

```
# Base Images## 从天池基础镜像构建 FROM registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:latest-cuda10.0-py3## 把当前文件夹里的文件构建到镜像的根目录下（.后面有空格，不能直接跟/）ADD . /## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）WORKDIR /## Install Requirements（requirements.txt包含python包的版本）## 这里使用清华镜像加速安装RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pipRUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt## 镜像启动后统一执行 sh run.shCMD ["sh", "run.sh"]
```

### 5\. 其它

- 按照官方要求把所需的文件全部按要求准备好即可。

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

03

线上提交

  
在所有的文件都准备之后，下面一步就是进行线上的提交，这里又分为三块。

1.  按照要求进行线上配置
2.  进行build和pull；
3.  提交,等待好消息;

### 1\. 按照要求进行线上配置

  
![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnETiaRbPCic3ZQNhYL3Fgicw7P1EFfjeMZJTX5L2UMyXlApYiaoEic7pLpxpg/640?wx_fmt=jpeg)  
![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnEYnS8QFqx2EaIj9T8tAhLLiaAQlhGjTqIicr4QCcW6qbIiaM9RRN2u30QQ/640?wx_fmt=jpeg)

### 2\. 进行build和pull\(按照自己的机器的操作系统进行docker的安装并在命令行操作\)

![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnE0dgGoVnux3qYNiarxYNpibiaYAJsR7yKUa3bkkH6fLTrpPLLvd7zWESCA/640?wx_fmt=jpeg)  
![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnEvNp9pNvdYd72gxoBcXvofqRm6KlpCMopTWVCRr1VHcYOiaAO5NAuHmA/640?wx_fmt=jpeg)  
![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnESoECd9mCibJ1cn6K6fE319EEgZGXG41roanDAzlYgmoAYoyDJYlUDfg/640?wx_fmt=jpeg)  

```
#### 1.登录sudo docker login --username="自己的用户名" registry.cn-shenzhen.aliyuncs.com#### 2.builddocker build registry.cn-shenzhen.aliyuncs.com/ai_earth_baseline/test_ai_earth_submit:1.0 .#### 3.pushdocker push registry.cn-shenzhen.aliyuncs.com/ai_earth_baseline/test_ai_earth_submit:1.0
```

### 3\. 提交

![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnEwqDQDT1j7WBtPTvT8Gp8kCnN8RzRm5JvfPNma44P3AsJdqdOBdL5qQ/640?wx_fmt=jpeg)

根据自己的不同进行提交即可，如果不出意外，等待一会儿，线上跑完了就会有结果了。

![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8fib3qBx5Q6WmmCxOtSSveGnEic95uxYwUaf5az8DItFf2wFicIu8XKFem8Oq5MhlzAcaq4CTjcWr24NQ/640?wx_fmt=jpeg)