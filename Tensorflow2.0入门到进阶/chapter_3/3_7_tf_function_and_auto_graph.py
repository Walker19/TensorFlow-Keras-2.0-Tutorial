# 本节一方面讲述如何将普通函数转化为TensorFlow中的图结构，
# 使得函数计算速度更快，另一方面讲述如何指定参数的形状或类型
# 最后还讲述了如何查看TensorFlow图函数的内部结构
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


var = tf.Variable(0.)


@tf.function
def add_21():
    # tf.function内部不能定义variable !
    return var.assign_add(21)  # +=


print(add_21())

# 通过input_signature(签名函数)限定输入参数的形状和类型，
# 如果不是指定的输入格式就会报错
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)

try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)

print(cube(tf.constant([1, 2, 3])))

# @tf.function: py func -> tf graph
# get_concrete_function -> add input signature -> SavedModel
# tf function 调用 get_concrete_function 后即可保存为 SavedModel

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32)
)

print(cube_func_int32)

print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1, 2, 3])))
# cube.get_concrete_function(tf.TensorSpec([5], tf.int32))
# True
# 说明两者的signature是一样的

# 仔细考察TensorFlow图结构的内部
cube_func_int32.graph.get_operations()
# 打印出cube_func_int32这个图函数的内部结构，从输入到操作到输出

pow_op = cube_func_int32.graph.get_operations()[2]  # 第三步，幂计算
print(pow_op)

print(list(pow_op.inputs))  # 图结构的输入
print(list(pow_op.outputs))  # 图结构的输出

cube_func_int32.graph.get_operation_by_name("x")  # 取出该图结构的变量 x
# x是 <tf.Operation 'x' type=Placeholder>
# 说明x是占位符，因为TensorFlow是静态图模式，先确定好输入的shape，再有数据再具体计算
# 所以使用的是占位符模式

cube_func_int32.graph.get_tensor_by_name("x:0")
# get_tensor_by_name，取出具体的值

cube_func_int32.graph.as_graph_def()
# TensorFlow图结构中操作和变量都是节点

