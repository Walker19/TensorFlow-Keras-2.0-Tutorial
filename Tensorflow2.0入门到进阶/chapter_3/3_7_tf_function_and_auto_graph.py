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


# tf.function and autograph
# tf.function可以将普通的python函数编译为TensorFlow中的计算图
# autograph是tf.function将普通py函数转为tf图的一种机制

# -> python function
def scaled_elu(z, scale=1.0, alpha=1.0):
    # 这个函数虽然内部使用tf函数，但是还是一个python函数
    # z >= 0? scale * z or scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2])))

# -> 方法 1. convert py function to tf graph using tf.function(xxx)
scaled_elu_tf = tf.function(scaled_elu)

print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.])))

# 调用.python_function即可获得tf图函数的python版本
print(scaled_elu_tf.python_function is scaled_elu)


# 之所以转化为tf中的图模式，是因为计算速度快，e.g:
# %timeit scaled_elu(tf.random.normal((1000, 1000)))  # 5 ms
# %timeit scaled_elu_tf(tf.random.normal((1000, 1000)))  # 1.12 ms

# -> 方法 2. convert py function to tf graph using 装饰器， e.g:
# 1 + 1/2 + 1/2^2 + 1/2^3 + ... + 1/2^n
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total


print(converge_to_2(20))


# 转化为图结构的中间代码实际上就是如下过程(源码),
def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    # work in jupyter
    display(Markdown('```python\n{}\n```'.format(code)))


display_tf_code(scaled_elu)
# continue 3-8 4:12
