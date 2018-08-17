# -*- coding:utf-8 -*- 
__Author__ = "Feiyang Chen"
__Time__ = '2018/8/17'
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

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
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

model = Sequential()
model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=input_shape, padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(features.train_X, features.train_Y, batch_size=batch_size, verbose=1,
          epochs=nb_epochs, validation_data=(features.test_X, features.test_Y))

print("\nTesting ...")
score, accuracy = model.evaluate(features.test_X, features.test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
