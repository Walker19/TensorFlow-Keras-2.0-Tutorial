# 本节主要讲述wide&deep模型的实战，实战数据是房价回归数据
# 本节重点讲述子类API的模型搭建方法
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
# 使用子类API

class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型的层次"""
        self.hidden1_layer = keras.layers.Dense(30, activation="relu")
        self.hidden2_layer = keras.layers.Dense(30, activation="relu")
        self.output_layer = keras.layers.Dense(1)

    def call(self, input):
        """完成模型的前向计算"""
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output


# 1 方法一，实例化
model = WideDeepModel()
# 2 方法二，构造为Sequential模型（我个人喜欢这种）
"""
model = keras.models.Sequential([
    WideDeepModel(),
])
"""
model.build(input_shape=(None, 8))  # build主要是为了告诉模型第一层输入的shape是怎样的
# 子类模型需要使用build方法才能使用如下，
# 如compile和fit等方法

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
