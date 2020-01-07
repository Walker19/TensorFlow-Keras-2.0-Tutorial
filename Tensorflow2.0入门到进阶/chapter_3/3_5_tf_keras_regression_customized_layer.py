# 本节主要讲述如何使用子类api模式自定义网络层
# 还讲述了如何使用lambda来自定义不需定义参数的网络层
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

# 常规dense layer的定义方法
layer = tf.keras.layers.Dense(100)  # 不指定输入shape,100是输出维度
layer = tf.keras.layers.Dense(100, input_shape=(None, 5))  # 指定shape
layer(tf.zeros([10, 5]))

layer.variables  # -> w
# x * w + b
layer.trainable_variables

help(layer)  # 查看层的说明

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


# 1 自定义层，使用子类api模式
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)  # 为什么？

    def build(self, input_shape):
        """构建所需要的参数"""
        # x * w + b;input_shape:[None, a]; w:[a, b]; output_shape:[None, b]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),  # 指定参数w的形状
                                      initializer='uniform',  # 参数使用均匀分布初始化
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)  # 为什么？

    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)


# 2 对于不想自定义参数的场景，可以使用lambda函数来定义网络层。e.g：
# 实现tf.nn.softplus: log(1+e^x)，这个东西就是一个激活函数层
customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))

print(customized_softplus([-10., -5., 0., 5., 10.]))

model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu',
                         input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # customized_softplus激活函数等价于以下两种方式:
    # keras.layers.Dense(1, activation="softplus")
    # keras.layers.Dense(1), keras.layers.Activation("softplus"),
])
model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd",  # 使用自定义损失函数
              metrics=["mean_squared_error"])
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

history = model.fit(x_train_scaled, y_train,
                    validation_data=(x_valid_scaled, y_valid),
                    epochs=100,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)
# ret = model.evaluate(x_test_scaled, y_test)
# print(ret)  # loss and acc
