# -*- coding: utf-8 -*-
"""
Description :   
     Author :   Yang
       Date :   2020/3/15
"""
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

# tfrecord是TensorFlow开发的一种文件格式，用于tf模型加快数据的读取和使用
# 能够有效减轻显存不够的压力
# 组成结构如下
# tfrecord(所有数据集)
# -> tf.train.Example(单个数据)
#       -> tf.train.Features -> {"key": tf.train.Feature}(每个数据是个字典，key-value是字段名和字段值，label也存在这里)
#             -> tf.train.Feature -> tf.train.ByteList(存储字符串)，FloatList, Int64List(这里是存储具体的数据结构)

# 1. 为了用tfrecord存储，TensorFlow规定需要先用tf.train.Example对
# 数据进行封装，然后序列化后才能存储
favorite_books = [name.encode('utf-8') for name in ["machine learning", "cc150"]]  # 字符串转化为utf-8格式
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)

hours_floatlist = tf.train.FloatList(value=[15.5, 9.5, 7.0, 8.0])
print(hours_floatlist)

age_int64list = tf.train.Int64List(value=[42])
print(age_int64list)

features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(bytes_list=favorite_books_bytelist),
        "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list),
    }
)  # 该数据总共有3个feature(特征)

print(features)

example = tf.train.Example(features=features)
print(example)

# 将内容序列化，减少空间
serialized_example = example.SerializeToString()
print(serialized_example)

# 2. 将序列化的内容保存到本地文件，以tfrecord格式存储
output_dir = 'tfrecord'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = "test.tfrecords"
filename_fullpath = os.path.join(output_dir, filename)

with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

# 3.使用tfrecord文件：将文件读取到Dataset
dataset = tf.data.TFRecordDataset([filename_fullpath])  # 只需一个路径就可以由tfrecord构建为dataset
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)  # 读取到的文件是序列化后的二进制代码

# 4.解析tfrecord读取后的字符串，使其成为原数据
expected_features = {
    "favorite_books": tf.io.VarLenFeature(dtype=tf.string),
    "hours": tf.io.VarLenFeature(dtype=tf.float32),
    "age": tf.io.FixedLenFeature([], dtype=tf.int64),
}

dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features
    )  # 将序列化字符串读取为tensor，得到sparsetensor
    # 将sparsetensor转变为普通tensor
    books = tf.sparse.to_dense(example['favorite_books'],
                               default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))

# 将数据进行压缩存储
filename_fullpath_zip = filename_fullpath + '.zip'
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip, options=options) as writer:
    for i in range(3):
        writer.write(serialized_example)

# 读取压缩的tfrecord文件
dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip],
                                      compression_type="GZIP")
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features
    )  # 将序列化字符串读取为tensor，得到sparsetensor
    # 将sparsetensor转变为普通tensor
    books = tf.sparse.to_dense(example['favorite_books'],
                               default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))

# continue https://www.bilibili.com/video/av95701611?p=48
