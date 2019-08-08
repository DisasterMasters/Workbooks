
# nn model
# Zhenning Yang
# Last modified: 08/08/2019

'''
Update Note:
    model:
        1. fixed predict_topk for binary classification

    helper functions:
        1. added get_sub_class_df
        2. added downsampling

TO_DO:
    1. add slim-bert to nn model

PS:
    1. might need to cite the source when using crawl_embeddings
'''

#
#
#

##########################################################################

import io

import numpy as np
import pandas as pd

import pickle

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras import regularizers
from keras.layers.convolutional import *
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.utils import class_weight

from time import time
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from keras_self_attention import SeqSelfAttention
except:
    pass

#a = 10
loss_functions = ['mean_squared_error', 'logcosh', 'mean_absolute_error', 'categorical_crossentropy', 'binary_crossentropy']
act = ['sigmoid', 'softmax']
opt = ['sgd', 'adam']

class keras_model:

    def __init__(self, df=None, text=None, label=None, test_split=0.10, lr=0.01,
                 num_words=10000, embedding_dim=50, dense_nodes=None, maxlen=None,
                 tb_dir=None, loss_func='categorical_crossentropy', pre_trained_EM=None,
                 test_df=None, cnn=False, act='softmax', lstm=True, opt='adam', drop=False):

        '''
        num_words: is the max limit of words
        test_df: expect to be a list. Ex. [df, text_label, code_label]
        if tb_dir: is not none, will create a dir
        pre_trained_EM: pass in a dict of word embeddings
        embedding_dim: if pass in a pre trained EM, embedding_dim might needs to be reset
        '''

        if isinstance(df, pd.DataFrame) and text!=None and label!=None:

            #print('var that outside the class: {}'.format(a))

            self.version = '6.0'

            self.df = df
            self.text = text #column name for text
            self.label = label #column name for label
            self.test_split = test_split
            self.classes = np.unique(df[label].values)

            pre_train = False

            self.opt = opt
            self.lr = lr
            self.maxlen = maxlen
            self.num_words = num_words
            self.labels_not_match = False

            self.loss_func = loss_func
            if (len(self.classes) == 2) and (self.loss_func=='categorical_crossentropy'):
                self.loss_func = 'binary_crossentropy'

            #visualization on TensorBoard
            self.tb_dir = tb_dir

            sentences = df[text].values
            encoder = LabelBinarizer()
            y = encoder.fit_transform(df[label].values)

            if test_df == None:
                # train test split & tokenization
                self.sentences_train, self.sentences_test, self.y_train, self.y_test = train_test_split(
                   sentences, y, test_size=self.test_split, random_state=42)
                self.tokenizer = Tokenizer(num_words=self.num_words)
                self.tokenizer.fit_on_texts(self.sentences_train)
                self.X_train = self.tokenizer.texts_to_sequences(self.sentences_train)
                self.X_test = self.tokenizer.texts_to_sequences(self.sentences_test)
                #vocab_size = len(self.tokenizer.word_index) + 1
            else:
                '''
                if user pass in a separate validation data set
                format: test_df = [df, text, label]
                '''
                # over write test_split -> 0
                self.test_split = 0

                self.valid_df = test_df[0]
                self.valid_text = test_df[1]
                self.valid_label = test_df[2]
                self.test_label = np.unique(self.valid_df[self.valid_label].values)
                if not np.array_equal(self.classes, self.test_label):
                    print('labels not match:')
                    self.labels_not_match = True
                    print('training set: {}'.format(self.classes))
                    print('testing set:  {}'.format(np.unique(self.valid_df[self.valid_label].values)))
                    #raise Exception('label not match')

                self.sentences_train = sentences
                self.y_train = y
                self.sentences_test = self.valid_df[self.valid_text].values
                self.y_test = encoder.transform(self.valid_df[self.valid_label].values)
                self.tokenizer = Tokenizer(num_words=self.num_words)
                self.tokenizer.fit_on_texts(self.sentences_train)
                self.X_train = self.tokenizer.texts_to_sequences(self.sentences_train)
                self.X_test = self.tokenizer.texts_to_sequences(self.sentences_test)

            self.vocab_size = len(self.tokenizer.word_index) + 1


            #maxlen_op = ['maxlen','mid','mean']
            if (type(maxlen) == int) or (type(maxlen) == float):
                self.maxlen = int(maxlen)
            else:
                #back to default
                #self.maxlen = 'maxlen'
                length_list = []
                for val in self.X_train:
                    length_list.append(len(val))
                self.maxlen = max(length_list)

            '''
            if type(self.maxlen) == str:
                length_list = []
                for val in self.X_train:
                    length_list.append(len(val))
                if self.maxlen == 'maxlen':
                    self.maxlen = max(length_list)
            '''

            self.X_train = pad_sequences(self.X_train, padding='post', maxlen=self.maxlen)
            self.X_test = pad_sequences(self.X_test, padding='post', maxlen=self.maxlen)

            # set pre trained embeddings
            if isinstance(pre_trained_EM, dict):
                # over write vocab_size
                # set pre_train = True
                NB_WORDS = len(pre_trained_EM)
                self.vocab_size = NB_WORDS
                pre_train = True

                missing_w = 0
                emb_matrix = np.zeros((NB_WORDS, embedding_dim))
                for w, i in self.tokenizer.word_index.items():
                    if i < NB_WORDS:
                        vect = pre_trained_EM.get(w)
                        if vect is not None:
                            emb_matrix[i] = vect
                        else:
                            missing_w += 1
                    else:
                        break

            '''
            Hyperparameter
            1. embedding_dim
            2. Dense nodes

            Different layers
            1. dropout layers
            '''

            self.embedding_dim = embedding_dim
            self.output_num = self.y_train.shape[1]

            if dense_nodes == None:
                self.dense_nodes = self.output_num * 10
            else:
                self.dense_nodes = dense_nodes

            #self.loss_func = ['mean_squared_error', 'logcosh', 'mean_absolute_error', 'categorical_crossentropy']

            '''
            1. em-de-de-output
            2. em-lstm-output
            3. em-cnn-de-output
            4. em-cnn-lstm-output
            5. note: an extra dropout layer can be added
            '''
            model = Sequential()
            #model.add(layers.Embedding(vocab_size, self.embedding_dim, input_length=self.maxlen))
            if pre_train:
                model.add(layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.maxlen, trainable=False))
                model.layers[0].set_weights([emb_matrix])
            else:
                model.add(layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.maxlen))

            if cnn:
                model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
                #model.add(layers.MaxPooling1D())
                if not lstm:
                    model.add(layers.MaxPooling1D())
                    model.add(layers.Dropout(0.3))
            if lstm:
                model.add(layers.LSTM(self.dense_nodes, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

            model.add(layers.Flatten())

            if drop:
                model.add(layers.Dropout(0.5))

            if (cnn and not lstm) or (not cnn and not lstm):
                model.add(layers.Dense(self.dense_nodes, activation='relu'))

            model.add(layers.Dense(self.output_num, activation=act))

            '''
            sgd = SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=False)
            adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            if self.opt == 'sgd':
                model.compile(loss=self.loss_func, optimizer=sgd, metrics=['accuracy'])
            else:
                self.opt = 'adam'
                model.compile(loss=self.loss_func, optimizer=adam, metrics=['accuracy'])

            '''
            self.model = model
            #self.model.summary()
        else:
            print("please load a model")


    def train(self, epochs=8, verbose=1, shuffle=True, batch_size=50, tb_name=time(), class_weight=None, opt=None, lr=None):

        '''
        self.epochs = epochs
        self.batch_size = batch_size
        '''
        if opt==None:
            opt=self.opt
        else:
            self.opt = opt

        if lr==None:
            lr=self.lr
        else:
            self.lr = lr

        sgd = SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=False)
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if self.opt == 'sgd':
            self.model.compile(loss=self.loss_func, optimizer=sgd, metrics=['accuracy'])
        else:
            self.opt = 'adam'
            self.model.compile(loss=self.loss_func, optimizer=adam, metrics=['accuracy'])

        val_data = None
        callback = None

        if not self.labels_not_match:
            val_data = (self.X_test, self.y_test)

        if self.tb_dir != None:
            tb = TensorBoard(log_dir='{}/{}'.format(self.tb_dir, tb_name), update_freq=50)
            callback = [tb]

        self.history = self.model.fit(self.X_train, self.y_train,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=val_data,
                    shuffle=shuffle, batch_size=batch_size, callbacks=callback, class_weight=class_weight)

        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        try:
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=False)
            print("Testing Accuracy:  {:.4f}".format(accuracy))
        except:
            print('testing labels not match')


    def plot_history(self):
        #plt.style.use('ggplot')
        acc = self.history.history['acc']
        loss = self.history.history['loss']

        try:
            val_acc = self.history.history['val_acc']
            val_loss = self.history.history['val_loss']
        except:
            pass

        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        try:
            plt.plot(x, val_acc, 'r', label='Validation acc')
        except:
            pass
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        try:
            plt.plot(x, val_loss, 'r', label='Validation loss')
        except:
            pass
        plt.title('Training and validation loss')
        plt.legend()


    def save_model(self, name):
        name = name + '.pkl'
        with open(name, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        print("model saved:  {}".format(name))



    def predict(self, text):
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post', maxlen=self.maxlen)
        rounded_predictions = self.model.predict_classes(text, verbose=0)
        predict = []
        for i in rounded_predictions:
            #test
            if isinstance(i, np.ndarray):
                i = i[0]
            #
            predict.append(self.classes[i])

        return predict


    def predict_topk(self, text, topk=2):
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post', maxlen=self.maxlen)
        predictions = self.model.predict(text, verbose=0)
        topk_list = []

        if len(self.classes) == 2:
            #temp = []
            for val in predictions:
                if val[0] < .5:
                    #print(val)
                    topk_list.append([self.classes[0], val[0]])
                else:
                    topk_list.append([self.classes[1], val[0]])
                #topk_list = temp
        else:
            for index in range(predictions.shape[0]):
                temp = []
                top = sorted(range(len(predictions[index])), key=lambda i: predictions[index][i], reverse=True)[:topk]
                #print('-'*20)
                for x in range(len(top)):
                    temp.append([self.classes[top[x]], predictions[index][top[x]]])
                    #print('{:3f} : {}'.format(predictions[index][top[x]], labels[top[x]]))

                topk_list.append(temp)

        return topk_list



    def infor(self):
        #self.df = df
        #self.text = text
        #self.label = label
        #self.test_split = test_split
        #self.classes = np.unique(df[label].values)
        #self.lr = lr
        #self.maxlen = maxlen
        #self.num_words = num_words
        sep_val = 'separate validation data'
        print('Model version: {}'.format(self.version))
        print('='*75)
        print('Detected labels:   {}'.format(self.classes))
        #print('test_split:        {}'.format(self.test_split))
        if self.test_split == 0:
            print('test_split:        {}'.format(sep_val))
            print('valid data labels: {}'.format(self.test_label))
        else:
            print('test_split:        {}'.format(self.test_split))

        print('num_words:         {}'.format(self.vocab_size))
        print('maxlen:            {}'.format(self.maxlen))
        print('Learning_Rate(lr): {}'.format(self.lr))
        print('loss function:     {}'.format(self.loss_func))
        print('='*75)
        #print('Binary Model: ')
        self.model.summary()



    def plot_matrix(self, normalize=True, size=(6,5), title=None):
        '''
        size = size
        normalize = normalize
        title = "Test set(" + str(self.test_split) + ")"

        #true_labels = [np.where(r==1)[0][0] for r in self.y_test]
        true_labels = self.y_test
        predictions = self.model.predict_classes(self.X_test, verbose=0)
        cnn_model.plot_confusion_matrix(true_labels, predictions, classes=self.classes, normalize=normalize, size=size, title=title)
        '''
        size = size
        normalize = normalize
        if title == None:
            if self.test_split == 0:
                title = 'separate validation data set'
            else:
                title = "Test set(" + str(self.test_split) + ")"
        try:
            true_labels = [np.where(r==1)[0][0] for r in self.y_test]
        except:
            true_labels = self.y_test
        predictions = self.model.predict_classes(self.X_test, verbose=0)
        plot_confusion_matrix(true_labels, predictions, classes=self.classes, normalize=normalize, size=size, title=title)





# ============================== helper functions ==============================
'''
functions:
1. save_obj
2. load_pkl
3. glove_to_dict
4. crawl_to_dict
5. oversampling
6. plot_confusion_matrix
7. eval
8. get_sub_class_df
9. downsampling
10. get_class_weight
'''


# save an object
def save_obj(obj, name):
    name = name + '.pkl'
    with open(name, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print("object saved:  {}".format(name))


#load a cnn class or any pkl object
def load_pkl(name):
    print("---------------------")
    print("loading")
    try:
        with open(name, 'rb') as obj:
            x = pickle.load(obj)

        print("---------------------")
        print("{} load".format(name))
        return x
    except:
        print("file not found")


# read a glove pre trained embeddings txt file
# and return a dictionary of word vectors
def glove_to_dict(path, dim):
    skip = 0
    glove_file = path
    emb_dict = {}
    glove = open(glove_file, encoding='utf-8')
    print('loading pre_trained Embeddings...\nfrom: {}'.format(path))
    for line in glove:
        values = line.split()
        word = values[0]
        try:
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) != dim:
                raise Exception()
        except:
            skip += 1
            continue
        emb_dict[word] = vector
    glove.close()
    print('{}\npre_trained Embeddings loaded'.format(35*'='))
    print('invalid word vectors: {}'.format(skip))

    return emb_dict


# load a crawl vec file
# return a dict
def crawl_to_dict(fname):
    print('loading crawl Embeddings...\nfrom: {}'.format(fname))
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    #print('vector file -> dict...')
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])

    print('map object  -> numpy array...')
    for key in data.keys():
        data[key] = np.asarray(list(data.get(key)))

    print('{}\npre_trained Embeddings loaded'.format(35*'='))
    return data



def oversampling(df, label, max_size=1000):
    print('oversampling to {}'.format(max_size))
    temp_df = df
    #max_size = 1000
    lst_two = [temp_df]
    for class_index, group in temp_df.groupby(label):
        lst_two.append(group.sample(max_size-len(group), replace=True))
    balance_df = pd.concat(lst_two)

    return balance_df



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, y_label='True label', x_label='Predicted label', size=(6,5)):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel=y_label,
           xlabel=x_label)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# evaluate the model
# average: binary, micro, macro, weighted and samples
def eval(y_true, pred, average='binary'):
    dict = {}
    dict['accuracy'] = accuracy_score(y_true, pred)
    dict['f1_score'] = f1_score(y_true, pred, average=average)
    dict['recall'] = recall_score(y_true, pred, average=average)
    dict['precision'] = precision_score(y_true, pred, average=average)

    return dict



# get a df with sub classes and all other classes labeled as -1/
# for building sub models
# sub_class is a list of labels
# code_label is the name of the column in DataFrame
# return a new df
def get_sub_class_df(df, sub_class, code_label, label=-1):
    labels = np.unique(df[code_label])
    temp = df.copy()
    for val in labels:
        if val not in sub_class:
            temp.replace({code_label: {val:label}}, inplace=True)

    return temp


# downsampling
# num: min number. default is the number of the smallest classes
# down_label: default apply downsample to all classes. (Optional: pass in a list of classes)
# return a new df
def downsampling(df, code_label, num=None, down_label=None):
    frame = []
    labels = np.unique(df[code_label])
    min_num = min(df[code_label].value_counts())

    def get_sample(frame, df, code_label, val, num, min_num):
        try:
            frame.append(df[df[code_label]==val].sample(num))
        except:
            frame.append(df[df[code_label]==val].sample(min_num))

        return frame

    if num == None:
        num = min_num

    if down_label == None:
        for val in labels:
            frame = get_sample(frame, df, code_label, val, num, min_num)
    else:
        for l in down_label:
            try:
                frame = get_sample(frame, df, code_label, l, num, min_num)
            except:
                pass

        for x in labels:
            if x not in down_label:
                frame.append(df[df[code_label]==x])

    temp = pd.concat(frame)
    return temp


# get class weights, based on the distribution of each classes in data
def get_class_weight(df, code_label):
    labels = np.unique(df[code_label])
    class_weights = class_weight.compute_class_weight('balanced', np.unique(df[code_label]), df[code_label])
    class_weights_dict = {}
    for i in range(len(class_weights)):
        class_weights_dict[labels[i]] = class_weights[i]

    return class_weights_dict
