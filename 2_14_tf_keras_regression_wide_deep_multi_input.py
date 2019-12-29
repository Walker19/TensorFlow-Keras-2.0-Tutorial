# 本节主要讲述wide&deep模型的实战，实战数据是房价回归数据
# 本节重点讲述模型多个输入的情况，实战案例将输入分成了两部分(将8个特征分成了两部分)，
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
# 多输入

input_wide = keras.layers.Input(shape=[5])  # 第一部分的输入
input_deep = keras.layers.Input(shape=[6])  # 第二部分的输入
hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_wide, input_deep],
                           outputs=[output])

print(model.summary())

model.compile(loss="mean_squared_error", optimizer="sgd")

print('---model construct over---')

x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]

x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]

x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]

callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)
]

# x_train_scaled_wide shape:[None, 5] -> input_wide
# x_train_scaled_deep shape:[None, 6] -> input_deep
history = model.fit(x=[x_train_scaled_wide, x_train_scaled_deep],  # 注意，由于自定义了输入(input)的shape，
                    # 所以会根据fit中输入的数据自动寻找满足shape的输入
                    y=y_train,
                    epochs=10,
                    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep],
                                     y_valid),
                    callbacks=callbacks)


# 保存模型训练的历史信息，用于打印输出，方便debug


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

ret = model.evaluate([x_test_scaled_wide, x_test_scaled_deep],
                     y_test)
print(ret)  # loss and acc
