# coding=utf-8
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
# from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD
import numpy as np
import keras
import matplotlib.pyplot as plt
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
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt: object, nb_filter: object, kernel_size: object, strides: object = (1, 1), with_conv_shortcut: object = False) -> object:
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

inpt = Input(shape=input_shape)
x = ZeroPadding2D((3, 3))(inpt)
x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(2, activation='softmax')(x)

model = Model(inputs=inpt, outputs=x)
sgd = SGD(decay=0.0001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(features.train_X, features.train_Y, batch_size=batch_size, verbose=1,
          epochs=nb_epochs, validation_data=(features.test_X, features.test_Y))

print("\nTesting ...")
score, accuracy = model.evaluate(features.test_X, features.test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)






