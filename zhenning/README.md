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

# check model information
model.infor()

# train
model.train(batch_size=50, epochs=8)

# plot history & confusion matrix
model.plot_history()

model.plot_matrix(size=(6,5))
```
