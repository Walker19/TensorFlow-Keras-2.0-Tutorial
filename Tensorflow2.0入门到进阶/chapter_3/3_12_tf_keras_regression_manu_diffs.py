# 本节主要讲述如何手动实现数据的批量训练全过程
# 包括遍历epoch，遍历batch，计算损失函数，计算梯度，更新参数以及计算评价指标等。
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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # 默认对每一列标准化
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])

model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd",
              metrics=["mean_squared_error"])

# metric 使用
metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))  # 9
print(metric([0.], [1.]))  # (9 + 1) / 2 = 5, 自动累加并平均了
print(metric.result())

# 重置之后就不会累加了
metric.reset_states()
metric([1.], [3.])
print(metric.result())  # 4

# 定义训练参数
epochs = 100
batch_size = 32
steps_per_epoch = len(x_train_scaled) // batch_size

optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()


def random_batch(x, y, batch_size=32):
    "随机取出数据训练"
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]


# fit里面都做了什么事情
# 1.batch 遍历训练集，计算评价指标 metric
#    1.1 自动求导
# 2.epoch结束 验证集 metric

# 手动求导训练，替代 fit 函数，效果其实差不多
for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_per_epoch):
        x_batch, y_batch = random_batch(x_train_scaled, y_train,
                                        batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, y_pred))
            # loss是当前batch的损失函数
            metric(y_batch, y_pred)  # 累积的评价指标效果
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradients(grads_and_vars)  # 梯度下降，更新参数
        print("\rEpoch", epoch, " train mse:",
              metric.result().numpy(), end=" ")
    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\t", "valid mse: ", valid_loss.numpy())
