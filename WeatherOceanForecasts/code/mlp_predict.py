import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input 
import numpy as np
import os
import zipfile

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def build_model():  
    inp    = Input(shape=(12,24,72,4))  
    
    x_4    = Dense(1, activation='relu')(inp)   
    x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))
    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))
    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))
     
    x = Dense(64, activation='relu')(x_1)  
    x = Dropout(0.25)(x) 
    x = Dense(32, activation='relu')(x)   
    x = Dropout(0.25)(x)  
    output = Dense(24, activation='linear')(x)   
    model  = Model(inputs=inp, outputs=output)

    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99) 
    model.compile(optimizer=adam, loss=RMSE)

    return model 

model = build_model()
model.load_weights('./user_data/model_data/model_mlp_baseline.h5')

test_path = './tcdata/enso_round1_test_20210201/'

### 1. 测试数据读取
files = os.listdir(test_path)
test_feas_dict = {}
for file in files:
    test_feas_dict[file] = np.load(test_path + file)
    
### 2. 结果预测
test_predicts_dict = {}
for file_name,val in test_feas_dict.items():
    test_predicts_dict[file_name] = model.predict(val).reshape(-1,)
#     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])

### 3.存储预测结果
for file_name,val in test_predicts_dict.items(): 
    np.save('./result/' + file_name,val)

#打包目录为zip文件（未压缩）
def make_zip(source_dir='./result/', output_filename = 'result.zip'):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    print(source_dirs)
    for parent, dirnames, filenames in source_dirs:
        print(parent, dirnames)
        for filename in filenames:
            if '.npy' not in filename:
                continue
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
make_zip()
