# 建立深度网络模型

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D, Flatten

def get_model(shape):
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', input_shape=shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')
    return model
