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
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Bidirectional, Dropout, TimeDistributed, BatchNormalization
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from Extract_feature_2_class import Extract_feature  # local python class with Audio feature extraction (librosa)
#from Extract_Spectrum_Feature_2_class import Extract_Spectrum_feature
import matplotlib.pyplot as plt
from keras.models import Model
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"


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
#features.load_preprocess_data()
features.load_deserialize_data()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
#opt = Adam(0.00001)
#opt =Adadelta(0.01)
#opt = RMSprop(0.00001)
opt = SGD(0.001)

batch_size = 20
nb_epochs = 100
#--------------------------adam:
#batch_size =20 0.000001 5lun  62%
#batch_size =20 0.00001  17lun 68.78%
#batch_size =20 0.1  5lun   68.61%--->31.39%

#batch_size =10 0.000001  20lun 57%

#batch_size =15 0.00001  10lun 68.61%

#batch_size =10 0.00001  12lun 69.64   16lun 68.61    17lun 68.44     19lun 68.78  (best)

#batch_size =5 0.00001  6lun 67.24     11lun 68.95    13lun 68.61

#batch_size = 30   0.0001  7lun  68.61%
#batch_size = 10   0.0001  3lun 68.61%
#-----------------adadelta:
#batch_size =20 0.00001  10lun  40%
#batch_size =10 0.01  40lun  68.10%   52lun 68.95%    65lun  68.61%  (good)

#-------------------RMSprop:
#batch_size =20 0.001  1lun 68.61
#batch_size =20 0.00001  6lun 66.38%  10lun 68.61%

print("Training X shape: " + str(features.train_X.shape))
print("Training Y shape: " + str(features.train_Y.shape))


print("Test X shape: " + str(features.test_X.shape))
print("Test Y shape: " + str(features.test_Y.shape))

print("debug")



input_shape = (features.train_X.shape[1], features.train_X.shape[2])
print('Build model ...')
input_data = Input(shape=(features.train_X.shape[1], features.train_X.shape[2]))
print("--------input_data OK-----------")
BN = BatchNormalization()(input_data)
print("--------BN OK--------")
lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.35))(BN)
print("--------lstm OK-----------")
inter = Dropout(0.9)(lstm)
print("--------inter OK-----------")
inter1 = TimeDistributed(Dense(200, activation='relu'))(inter)
print("--------inter1 OK-----------")
lstm2 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.4, recurrent_dropout=0.35))(inter1)
print("--------lstm2 OK-----------")
inter2 = Dropout(0.9)(lstm2)
print("--------inter2 OK-----------")
output = Dense(units=features.train_Y.shape[1], activation='sigmoid')(inter2)
print("--------output OK-----------")


model = Model(input_data, output)
with tf.device("/gpu:0"):
    print("Compiling ...")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()




    print("Training ...")
    model.fit(features.train_X, features.train_Y, batch_size=batch_size, verbose=1,
          epochs=nb_epochs, shuffle=True, validation_data=(features.test_X, features.test_Y),class_weight='auto' )

print("\nTesting ...")
score, accuracy = model.evaluate(features.test_X, features.test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

history = LossHistory()
# acc-loss graph
history.loss_plot('epoch')

