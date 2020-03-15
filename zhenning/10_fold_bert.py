from __future__ import absolute_import, division, print_function, unicode_literals
#!/usr/bin/env python
# coding: utf-8


# RUN
# python3 prog xlsx output_dir

import os
import sys
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd

#import nvidia_smi
from gpuinfo import GPUInfo

def gpu_usage():
    total_memory = 32000
    available_device = GPUInfo.check_empty()
    percent,memory=GPUInfo.gpu_usage()
    for i in range(len(memory)):
        memory[i] = float(memory[i])/total_memory 
    print(memory)
    return memory

def get_a_free_gpu():
    count = 0
    #os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"
    memory = gpu_usage()
    for i in range(len(memory)):
        if (memory[i]<.1):
            print('run on device {}, Usage: {:.2f}%'.format(i, memory[i]*100))
            break
    os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(i)

# helper functions
def load_data(tokenizer, df, text_col, code_col, SEQ_LEN):
    indices, sentiments = [], []
    for x in range(len(df)):
        ids, temp = tokenizer.encode(df[text_col][x], max_len=SEQ_LEN)
        indices.append(ids)
        sentiments.append(df[code_col][x])
        #print(text)
        #print(sentiments)
        #print(ids)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(sentiments)


def build(model, num, lr=0.00002):
    # @title Build Custom Model
    from tensorflow.python import keras
    from keras_bert import AdamWarmup, calc_train_steps

    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs = keras.layers.Dense(units=len(le.classes_), activation='softmax')(dense)

    decay_steps, warmup_steps = calc_train_steps(
        num,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model = keras.models.Model(inputs, outputs)

    for x in range(len(model.layers)):
        #print(x)
        model.layers[x].trainable = True

    ''' 
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True
    model.layers[-5].trainable = True
    model.layers[-6].trainable = True
    model.layers[-7].trainable = True
    '''
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True

    model.compile(
        AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model

# main
################################################################
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.00002

pretrained_path = '../../twitter/zhenning/Content_Analysis/NN/uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'
# set visable gpus
# os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"
# gpu_usage()
get_a_free_gpu()
# sys.exit()

# @title Load Basic Model
import codecs
from keras_bert import load_trained_model_from_checkpoint

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

# ============================================= read xlsx file ================================================
# ======================== enter col name for label
# ======================== enter col name for Text
import sys
df = pd.read_excel(sys.argv[1])
print("col name: ", end=" ")
code_label = input()
print("text name: ", end=" ")
text_label = input()
print('count: {}'.format(len(df)))
print(df[code_label].value_counts())

# @title Convert Data to Array
import numpy as np
from keras_bert import Tokenizer
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer(token_dict)

df.reset_index(drop=True, inplace=True)
df_x, df_y = load_data(tokenizer, df, text_label, code_label, SEQ_LEN)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df_y)

print('classes: {}'.format(le.classes_))

# print out available GPUs
from tensorflow.python.client import device_lib
import tensorflow as tf
print()
#print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print()

'''
# Limiting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold as sk

numFold = 10
kf = sk(n_splits=numFold, random_state=None, shuffle=True)
kf.get_n_splits(df[text_label], df[code_label].values)
i = 0
for train_index, test_index in kf.split(df[text_label], list(df[code_label])):
    print('Fold {}'.format(i))
    X_train, X_test = df[text_label][train_index].dropna(), df[text_label][test_index].dropna()
    y_train, y_test = df[code_label][train_index].dropna(), df[code_label][test_index].dropna()
    print('tweets:', len(X_train), len(X_test))
    print('labels:', len(y_train), len(y_test))
    
    train_df = X_train.to_frame()
    train_df[code_label] = y_train
    test_df = X_test.to_frame()
    test_df[code_label] = y_test
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_x, train_y = load_data(tokenizer, train_df, text_label, code_label, SEQ_LEN)
    test_x, test_y = load_data(tokenizer, test_df, text_label, code_label, SEQ_LEN)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    #test 
    #print(train_df.head())

    print('loading weights...')
    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=True,
        trainable=True,
        seq_len=SEQ_LEN,
    )
    print('building')
    model = build(model, len(train_y), lr=LR)
    
    validation_data=(test_x, test_y)
    try:
        history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=validation_data)
    except:
        print('trainig error at fold_{}'.format(i))
    # save 
    #model.save('10fold_V4/weights/keras_bert_fold_V4_{}.h5'.format(i))
    # save excel file
    predictions = model.predict(train_x, verbose=1)
    topk_list = []
    topk = 1
    for index in range(predictions.shape[0]):
        temp = []
        top = sorted(range(len(predictions[index])), key=lambda i: predictions[index][i], reverse=True)[:topk]
        topk_list.extend(top)

    fold = pd.DataFrame(train_y, columns=['train_y'])
    fold['pred_train'] = topk_list
    fold['train_y'] = le.inverse_transform(fold['train_y'])
    fold['pred_train'] = le.inverse_transform(fold['pred_train'])
    fold['Text'] = train_df[text_label]
    

    predictions = model.predict(test_x, verbose=1)
    topk_list = []
    topk = 1
    for index in range(predictions.shape[0]):
        temp = []
        top = sorted(range(len(predictions[index])), key=lambda i: predictions[index][i], reverse=True)[:topk]
        topk_list.extend(top)

    foldt = pd.DataFrame(test_y, columns=['test_y'])
    foldt['pred_test'] = topk_list
    foldt['test_y'] = le.inverse_transform(foldt['test_y'])
    foldt['pred_test'] = le.inverse_transform(foldt['pred_test'])
    foldt['Text'] = test_df[text_label]

    with pd.ExcelWriter('{}/excel_files/fold_{}.xlsx'.format(sys.argv[2], i)) as writer:  # doctest: +SKIP
        fold.to_excel(writer, sheet_name='train')
        foldt.to_excel(writer, sheet_name='test')
    i+=1

