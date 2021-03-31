import tensorflow as tf
import numpy as np
import cv2
import os

from efficientnet_lite import efficientnet_lite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/127.5 - 1
x_test = x_test/127.5 - 1

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = efficientnet_lite(input_shape=(28, 28, 1), alpha=1)
model.summary()

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
    epochs=20,
    verbose=1
)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
