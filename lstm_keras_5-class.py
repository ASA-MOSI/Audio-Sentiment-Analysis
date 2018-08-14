# -*- coding:utf-8 -*- 
__Author__ = "Feiyang Chen"
__Time__ = '2018/8/14'
__Title__ = 'Audio_Sentiment_Analysis'

"""
 code is far away from bugs with the god animal protecting
    I love animals. God bless you.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃ 永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam

from Extract_feature_5_class import Extract_feature  # local python class with Audio feature extraction (librosa)
from keras.layers import Bidirectional
import matplotlib.pyplot as plt


np.random.seed(1234)


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
print("------Begin------")
features = Extract_feature()
features.load_preprocess_data()
# genre_features.load_deserialize_data()
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 35
nb_epochs = 10

print("Training X shape: " + str(features.train_X.shape))
print("Training Y shape: " + str(features.train_Y.shape))

print("Test X shape: " + str(features.test_X.shape))
print("Test Y shape: " + str(features.test_Y.shape))

input_shape = (features.train_X.shape[1], features.train_X.shape[2])
print('Build LSTM RNN model ...')
model = Sequential()

model.add(Bidirectional(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False)))
model.add(Dense(units=features.train_Y.shape[1], activation='softmax'))


print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# 创建一个实例history
history = LossHistory()

print("Training ...")
model.fit(features.train_X, features.train_Y, batch_size=batch_size, verbose=1,
          epochs=nb_epochs, validation_data=(features.test_X, features.test_Y), callbacks=[history])

print("\nTesting ...")
score, accuracy = model.evaluate(features.test_X, features.test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# 绘制acc-loss曲线
history.loss_plot('epoch')