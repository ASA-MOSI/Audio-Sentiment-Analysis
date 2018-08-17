# -*- coding:utf-8 -*- 
__Author__ = "Feiyang Chen"
__Time__ = '2018/8/16'
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

# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
import os
from Extract_Spectrum_Feature_2_class import Extract_Spectrum_feature
from keras.optimizers import Adam

seed = 1234
np.random.seed(seed)

# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
print("------Begin------")

features = Extract_Spectrum_feature()
# feature.load_preprocess_data()
#features = Extract_feature()
features.load_deserialize_data()

# features.load_deserialize_data()
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 2
nb_epochs = 5

print("Training X shape: " + str(features.train_X.shape))
print("Training Y shape: " + str(features.train_Y.shape))

print("Test X shape: " + str(features.test_X.shape))
print("Test Y shape: " + str(features.test_Y.shape))

input_shape = (features.train_X.shape[1], features.train_X.shape[2], features.train_X.shape[3])

# train_set, valid_set, test_set = cPickle.load(data)
# # train_x is [0,1]
# train_x = train_set[0].reshape((-1, 28, 28, 1))
# train_y = to_categorical(train_set[1])
#
# valid_x = valid_set[0].reshape((-1, 28, 28, 1))
# valid_y = to_categorical(valid_set[1])
#
# test_x = test_set[0].reshape((-1, 28, 28, 1))
# test_y = to_categorical(test_set[1])

model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(features.train_X, features.train_Y, validation_data=(features.test_X, features.test_Y), batch_size=20, epochs=20, verbose=2)

print(model.evaluate(features.test_X, features.test_Y, batch_size=20, verbose=2))