# 本节首先讲述初中知识如何对变量求导
# 然后讲述如何利用TensorFlow自动求导
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


# 初中知识是怎么求导的
def f(x):
    # 数学结论：df_dx = 6*x + 2
    return 3. * x ** 2 + 2. * x - 1


def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)


print(approximate_derivative(f, 1.))


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    # lambda x: g(x, x2) 变为关于x变量的一个函数，注意
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2


print(approximate_gradient(g, 2., 3.))

# 看看TensorFlow如何自动对变量求导：
# 使用方法一：(只能调用一次tape)
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)  # 求变量 z 对 x1 的偏导数
print(dz_x1)

try:
    dz_x2 = tape.gradient(z, x2)
    # 注意，tape只能被调用一次，求一次梯度就完啦
    # 调用两次就报错
except RuntimeError as ex:
    print(ex)

# 使用方法二：（多次调用tape）
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)

# persistent默认为false，一次调用就自动删除
# 所以第一种情况调用x2的偏导就报错，但是这里
# 设置不自动销毁tape变量，所以可以多次调用
dz_x1 = tape.gradient(z, x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1)
print(dz_x2)

del tape

# 使用方法三：（利用tape一次求多个变量的梯度）
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2)

# 如何对constant计算梯度？
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)

with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2)  # 这样不会得到梯度

x1 = tf.constant(2.0)
x2 = tf.constant(3.0)

with tf.GradientTape() as tape:
    # 手动添加
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2)

# 多个变量对一个变量求导，如何进行？
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
print(tape.gradient([z1, z2], x))  # 13
# 注意两个梯度直接相加了，不是分别输出


# 如何求二阶导？
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])  # dz_x1, dz_x2
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2])
               for inner_grad in inner_grads]
print(outer_tape)
del inner_tape
del outer_tape
# 输出：二维矩阵
# dz2_dx1x1, dz2_dx2x1, dz2_dx1x2, dz2_dx2x2


# 梯度下降的模拟：
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)  # 每次挪动一小步
print(x)

# 梯度下降和 keras optimizer 结合
learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr=learning_rate)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)  # z 可以理解为总的损失函数，x 是解释变量
    optimizer.apply_gradients([(dz_dx, x)])  # x 按照随机梯度下降计算公式更新参数
print(x)
