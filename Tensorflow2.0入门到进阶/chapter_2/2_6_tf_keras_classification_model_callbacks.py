# 本节主要讲述callbacks，它可以在模型训练中添加操作：
# 如，early stopping，
# resource：https://www.bilibili.com/video/av79196096?p=18
# docs:https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/keras/callbacks
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

# 对数据进行归一化
# x = (x - u) / std

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit_transform：在训练集计算均值和方差，并保存均值和方差以便测试集和验证集使
# 用相同的均值和方差，就直接使用transform
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

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

# 参数量的计算：
# flattened [None, 784] * W + b -> [None, 300]
# None is batch size
# W.shape = [784, 300], b.shape = [300,]
# so, 784*300+300=235500


# Tensorboard, earlystopping, ModelCheckpoint
logdir = 'callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                 "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),  # 模型运行状况
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True),  # 模型保存本地
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)  # 选择最优的模型保存
]

# tensorboard打开方法：在callbacks目录，conda cmd进入venv python，输入：tensorboard --logdir=callbacks
# tensorboard主要可以查看模型信息：包括loss、acc、图结构等

history = model.fit(x_train_scaled, y_train, epochs=20,
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
