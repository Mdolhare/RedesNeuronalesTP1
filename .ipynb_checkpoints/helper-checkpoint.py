import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow import keras

# Define a custom callback to compute the F1 macro score
class MetricsCallback(keras.callbacks.Callback):
    def __init__(self):
        super(MetricsCallback, self).__init__()
    def  on_train_begin(self,logs={}):
        self.f1_macro=[]
    def on_epoch_end(self, epoch, logs=None):
        y_pred=self.model.predict(testX).round()
        y_true=y_test
        score=f1_score(y_true, y_pred, average='macro')
        self.f1_macro.append(score)
        print(" F1 macro :",score)