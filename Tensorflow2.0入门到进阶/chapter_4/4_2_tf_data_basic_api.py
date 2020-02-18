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

# 3. interleave:对每个元素操作并返回
# 例子：文件名dataset -> 具体数据集

dataset2 = dataset.interleave(
    # map_fn

)