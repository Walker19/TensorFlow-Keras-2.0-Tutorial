# 本节从零开始搭建网络模型，并训练和测试
# resource：https://www.bilibili.com/video/av79196096?p=15
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")
    plt.show()


# show_single_image(x_train[0])

# construct model
# tf.keras.models.Sequential()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))  # 模型的第一层，功能是将28*28的输入矩阵拉平为向量
model.add(keras.layers.Dense(300, activation="sigmoid"))
model.add(keras.layers.Dense(100, activation="sigmoid"))
model.add(keras.layers.Dense(10, activation="softmax"))

# relu: y = max(0, x)
# softmax: 将向量变成概率分布。x = [x1, x2, x3]
#             y = [e^x1 / sum, e^x2 / sum, e^x3 / sum]
# 其中，sum = e^x1 + e^x2 + e^x3


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
# loss="sparse_categorical_crossentropy"的使用情景：
# 当标签y为类别中某个类别index时，而非one-hot向量时，应该这样。
# 如果y为one-hot，那么loss="categorical_crossentropy"

print('---model construct over---')

print(model.layers)  # 查看构建的图有哪些层
print(model.summary())

# 参数量的计算：
# flattened [None, 784] * W + b -> [None, 300]
# None is batch size
# W.shape = [784, 300], b.shape = [300,]
# so, 784*300+300=235500


history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_valid, y_valid))
# 保存模型训练的历史信息，用于打印输出，方便debug
print(type(history))  # 这是TensorFlow的callback

print(history.history)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)
