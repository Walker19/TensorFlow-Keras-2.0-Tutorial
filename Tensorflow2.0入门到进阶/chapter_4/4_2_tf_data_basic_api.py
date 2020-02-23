# 本章讲述数据的处理，极其重要
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

# 将内存数据构建为Dataset，list or np.array ...
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

# Dataset的操作
for item in dataset:
    print(item)

# Dataset常用操作
# 1. repeat -> epoch
# 2. get batch
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# 3. interleave:对"每个元素"操作并返回得到新的dataset
# 这里说的每个元素就是字面意思的每个元素，而不是从dataset中迭代取出的一批元素!注意
# 例子：文件名dataset -> 读取文件名，得到具体数据集 ->
# 得到大的新的数据集
# (文件名dataset是指：dataset的元素是个字符串，表示文件路径名)

dataset2 = dataset.interleave(
    # map_fn 对元素做的操作
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    # cycle_length 并行数量
    cycle_length=5,
    # block_length 从上面操作的元素中每次取出多少个
    block_length=5,
    # block_length 通过取出元素区块数目达到均匀混合(随机)的效果
)

for item in dataset2:
    print(item)

# 元祖构建 dataset
x = np.array([[1, 2, ], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])

dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

# 字典作为dataset的输入
dataset4 = tf.data.Dataset.from_tensor_slices({
    'feature': x,
    'label': y
})
# 可见 x, y 的长度需要匹配

for item in dataset4:
    print(item['feature'].numpy(), item['label'].numpy())
    # 每次取出x,y的各一个元素
