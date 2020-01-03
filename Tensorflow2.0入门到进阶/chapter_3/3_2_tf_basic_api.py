# 本节讲述 tf 的基础api，包括常量tensor，ragged tensor, sparse tensor, 变量的定义与使用
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

# index operation
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t)  # 2.0 默认打开 eager 模式，所以可以打印获取值，而不用tf.Session
print(t[:, 1:])
print(t[..., 1])

# ops operation
print(t + 10)
print(tf.square(t))
print(t @ tf.transpose(t))  # @: 矩阵相乘

# numpy convertion
print(t.numpy())
print(np.square(t))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(np_t))

# Scalars
t = tf.constant(2.718)
print(t.numpy())
print(t.shape)  # () -> 0维的tensor

# strings
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit="UTF*_CHAR"))
print(tf.strings.unicode_decode(t, "UTF8"))  # 转换为编码

# string array
t = tf.constant(["cafe", "coffee", "咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t, "UTF8")
print(r)  # 不等长的tensor -> tf.RaggedTensor -> 想象为 python list

r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [4]])
# index op on ragged tensor
print(r)
print(r[1])  # 第 2 行, 得到普通tensor
print(r[1:2])  # 第 2 行, 得到ragged tensor

# ops on ragged tensor
r2 = tf.ragged.constant([[51, 52], [], [71]])
print(tf.concat([r, r2], axis=0))  # 按行拼接，类似list.append

r3 = tf.ragged.constant([[13, 14], [15], [], [42, 43]])
print(tf.concat([r, r3], axis=1))  # 按列拼接需要保证行数相等

# convert ragged tensor to normal tensor, pad 0 if empty
print(r.to_tensor())

# 上面可以看到 to_tensor 时 0 的填充是在后面的，但是问题：如果矩阵大部分为0会有存储浪费

# sparse tensor：原理，记录矩阵中非零元素的索引，忽略为0元素的索引，
# 这样可以解决非常稀疏（矩阵中大部分元素为0）的矩阵的存储问题
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],  # 非零元素的索引, 需要按照 '顺序' 填写
                    values=[1., 2., 3.],  # 非零元素的值（按序）
                    dense_shape=[3, 4]  # 矩阵的原始维度
                    )
print(s)
print(tf.sparse.to_dense(s))

# ops on sparse tensors
s2 = s * 2.0
print(s2)

try:
    s3 = s + 1  # 不能加法
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])
print(tf.sparse.sparse_dense_matmul(s, s4))

# 如果构造sparse tensor时，索引不按照矩阵应有的顺序
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1], [2, 3]],  # 前两个索引颠倒顺序了
                     values=[1., 2., 3.],  # 非零元素的值（按序）
                     dense_shape=[3, 4]  # 矩阵的原始维度
                     )
print(s5)
s6 = tf.sparse.reorder(s5)  # 前两个索引颠倒顺序了，则重新排序即可to_dense
print(tf.sparse.to_dense(s6))

# Variables，可以更新变化的量
v = tf.Variable([[1., 2., 3.],
                 [4., 5., 6.]])
print(v)
print(v.value())
print(v.numpy())

# 变量的特点 -> 可以更新
# assign value
v.assign(2 * v)  # 含义： v = v * 2
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())

# 如果你想使用 = 代替 assign？ 做梦
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
