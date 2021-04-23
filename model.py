import os
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential #models.Sequential，用来一层一层一层的去建立神经层
from keras.layers import Dense, Activation #layers.Dense 意思是这个神经层是全连接层 layers.Activation 激励函数
from keras.optimizers import Adam

#建立卷积神经网络CNN模型
class_dim = 2 #总共两种声音，无人机与非无人机

#添加第一个卷积层，滤波器数量为32，大小是5*5，Padding方法是same即不改变数据的长度和宽带。
model = Sequential()
model.add(Convolution2D(
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',      # Padding method
    data_format='channels_last',
    input_shape=(40, 376, 1),
))
model.add(Activation('relu'))

#第一层 pooling（池化，下采样），分辨率长宽各降低一半，输出数据shape为（20，188, 32）
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_last',
))

#再添加第二卷积层，输出滤波器数量64，卷积核大小5
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))

#第二层池化层
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))

#经过以上处理后，数据shape为（10，94，64），需展开为一维，再连接全连接层
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#添加全连接层（输出层）
model.add(Dense(class_dim))
model.add(Activation('softmax'))

model.summary()

#编译，设置adam优化方法
adam = Adam(lr=1e-3)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.save(filepath='models/new_test.h5')
