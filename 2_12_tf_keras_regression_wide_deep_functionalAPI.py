# 本节主要讲述wide&deep模型的实战，实战数据是房价回归数据
# 本节重点讲述函数式API的构造
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

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit_transform：在训练集计算均值和方差，并保存均值和方差以便测试集和验证集使
# 用相同的均值和方差，就直接使用transform
x_train_scaled = scaler.fit_transform(x_train)  # 默认对每一列标准化
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)

# construct model
# 由于wide&deep模型并不是严格的层级结构（一层层add），所以这里并不使用Sequential方式搭建
# 这里使用函数式API、功能API
# Input 层，指定模型的输入shape
inputs = keras.layers.Input(shape=x_train.shape[1:])  # 对了，注意这里只用输入每个样本的输入维度，无须输入batch维度
hidden1 = keras.layers.Dense(30, activation="relu")(inputs)  # 这就是函数式API的特性，直接call即可
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# 函数式API效果类似复合函数：f(x) = h(g(x))

concat = keras.layers.concatenate([inputs, hidden2])
output = keras.layers.Dense(1)(concat)  # 回归问题，所以无须激活函数

model = keras.models.Model(inputs=[inputs],
                           outputs=[output])

print(model.summary())

model.compile(loss="mean_squared_error", optimizer="sgd")

print('---model construct over---')


callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)
]
history = model.fit(x_train_scaled, y_train, epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)


# 保存模型训练的历史信息，用于打印输出，方便debug


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

ret = model.evaluate(x_test_scaled, y_test)
print(ret)  # loss and acc
