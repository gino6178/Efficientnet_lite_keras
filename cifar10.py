import tensorflow as tf
import numpy as np
import cv2
import os

from efficientnet_lite import efficientnet_lite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)


cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/127.5 - 1
x_test = x_test/127.5 - 1

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = efficientnet_lite(input_shape=(32, 32, 3), alpha=1)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    batch_size=32,
    validation_data=(x_test, y_test),
    epochs=10000,
    verbose=1
)
