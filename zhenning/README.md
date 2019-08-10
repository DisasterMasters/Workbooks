# nn module
- keras_model
    - Feedforward Neural Network (FFNN)
    - Convolutional Neural Networks (CNN)
    - Attention model
    - Recurrent Neural Network (RNN)
        - LSTM (Default setting)
        - Self Attention based RNN
    - CNN + LSTM (Testing)
- Help functions
    - save_obj()
    - load_pkl()
    - glove_to_dict()
    - crawl_to_dict()
    - oversampling()
    - plot_confusion_matrix()
    - eval()
    - get_sub_class_df()
    - downsampling()
    - get_class_weight()

# Install
```
pip install keras 
```
### Self Attention based RNN "slim-bert"
```
pip install keras-self-attention
```

# Usage
```
import nn
from nn import keras_model

# default LSTM 
model = keras_model(dataFrame, text_col, label_col)
```

```
# check model information
model.infor()
--> Model version: 6.0
    ===========================================================================
    Detected labels:   [ 1  2  3  4  6  7  8  9 11 13 15 18 19]
    test_split:        0.1
    num_words:         6773
    maxlen:            56
    Learning_Rate(lr): 0.01
    loss function:     categorical_crossentropy
    ===========================================================================
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 56, 50)            338650    
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 56, 130)           94120     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 7280)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 13)                94653     
    =================================================================
    Total params: 527,423
    Trainable params: 527,423
    Non-trainable params: 0
    _________________________________________________________________
```

```
# train
model.train(batch_size=50, epochs=8)
--> Train on 2449 samples, validate on 273 samples
    Epoch 1/8
    2449/2449 [==============================] - 12s 5ms/step - loss: 2.0702 - acc: 0.3814 - val_loss: 1.4502 - val_acc:   
    0.5348

```
# plot history & confusion matrix
model.plot_history()

model.plot_matrix(size=(6,5))
```

# predit
model.predict(['How are you doing?', 'pass in a list of texts'])
--> [13, 13]

# 5. predict_topk
model.predict_topk(['How are you doing?', 'pass in a list of texts'], topk=2)
--> [[[13, 0.08009244], [11, 0.077788845]], 
     [[13, 0.08017371], [19, 0.07770648]]]
```
