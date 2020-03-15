#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import krippendorff

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, y_label='True label', x_label='Predicted label', size=(6,5), name=time()):

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
    plt.savefig('{}.png'.format(name))
    return ax


# In[2]:


df_list = []
df = pd.DataFrame()

for i in range(10):
    print(i)
    temp = pd.read_excel('excel_files/fold_{}.xlsx'.format(i), sheet_name='test')
    temp.drop(columns='Unnamed: 0', inplace=True)
    df_list.append(temp)
    df = df.append(temp)


# In[3]:


eval_report = {}
i = 0
for val in df_list:
    acc = accuracy_score(val['test_y'], val['pred_test'])
    print('fold {}  acc: {:.5}'.format(i, acc), end="")
    coder1 = val['test_y'].values
    coder2 = val['pred_test'].values
    k = krippendorff.alpha([coder1, coder2])
    print('  krippendorff: {:.5}'.format(k))
    i+=1
    
print('\navg acc: {}'.format(accuracy_score(df['test_y'], df['pred_test'])))
print('avg krp: {}'.format(krippendorff.alpha([df['test_y'].values, df['pred_test'].values])))


# In[4]:


a = classification_report(df['test_y'], df['pred_test'], output_dict=True)


# In[5]:


da = pd.DataFrame(a)
#da.head()

le = LabelEncoder()
y = le.fit_transform(df['test_y'])
pred = le.transform(df['pred_test'])


# In[12]:


plot_confusion_matrix(y, pred, le.classes_, size=(8,7), normalize=True, name='test_nor')
plot_confusion_matrix(y, pred, le.classes_, size=(8,7), normalize=False, name='test')

