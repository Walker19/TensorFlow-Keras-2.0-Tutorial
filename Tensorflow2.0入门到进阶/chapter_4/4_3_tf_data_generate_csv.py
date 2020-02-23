# 本节主要讲述如何利用 tf.data api从csv文件中读取数据并构造数据集
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

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

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

# 生成csv数据集
output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def save_to_csv(output_dir, data, name_prefix,
                header=None, n_parts=10):
    "将数据分为10个部分，然后写为csv文件。其中data为融合了特征和标签的信息"
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []

    for file_idx, row_indices in enumerate(
            np.array_split(np.arange(len(data)), n_parts)):

        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)

        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join(
                    [repr(col) for col in data[row_index]]))
                f.write('\n')

    return filenames


train_data = np.c_[x_train_scaled, y_train]  # 沿着列方向竖着扩充
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]

header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, "train",
                              header_str, n_parts=20)

valid_filenames = save_to_csv(output_dir, valid_data, "valid",
                              header_str, n_parts=10)

test_filenames = save_to_csv(output_dir, test_data, "test",
                             header_str, n_parts=20)

import pprint

print("train filenames:")
pprint.pprint(train_filenames)
print("valid filenames:")
pprint.pprint(valid_filenames)
print("test filenames:")
pprint.pprint(test_filenames)

# 如何将上述文件路径名列表读取为关于内容的 dataset，两个步骤：
# 1.filename -> dataset
# 2.read file ->(interleave) dataset -> datasets -> merge
# 3.parse 解析csv文件

# 第一步：转化为文件名 dataset
filename_dataset = tf.data.Dataset.list_files(train_filenames)
# list_files 专门将文件名转化为 dataset
for filename in filename_dataset:
    print(filename)

# 第二步，将文件名dataset读取为真实dataset
n_readers = 5
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    # TextLineDataset,按行读取文件内容
    # skip(1),跳过数据的第一行(标题栏)
    cycle_length=n_readers
)

for line in dataset.take(15):
    print(line.numpy())
    # '0.04049225382803661,-0.6890414153725881,-0.44379851741607473,
    # 0.022374585146687852,-0.22187226486997497,-0.1482850314959248,-0.8883662012710817,
    # 0.6366409215825501,2.852'
    # 这是csv的一行，分别展开的8个字段，最后个是label

# 第三步：解析csv，注意这里只能解析csv--->以,分割
# tf.io.decode_csv(str, record_defaults),str是需要转换的字符串，
# record_defaults是将该字段转换为特定数据类型
# 注意，我们这里是对csv每行(str)处理

# 示例
sample_str = '1,2,3,4,5'
# record_defaults 1:
# record_defaults = [tf.constant(0, dtype=tf.int32)] * 5

# record_defaults 2:
record_defaults = [
    tf.constant(0, dtype=tf.int32),
    0,
    np.nan,  # type(np.nan) == float
    'hello',
    tf.constant([])
]
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
# record_defaults中的变量只是作为参照，来将sample_str中的变量转为同样数据类型
print(parsed_fields)

# 如果传入错误字符串怎么办？
try:
    parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
    # parsed_fields2 = tf.io.decode_csv('1,2,3,4,5,6,7',  record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)


def parse_csv_line(line, n_fields):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0: -1])
    y = tf.stack(parsed_fields[-1:])
    return x, y


parse_csv_line(
    b'0.801544314532886,0.27216142415910205,-0.11624392696666119,-0.2023115137272354,-0.5430515742518128,-0.021039615516440048,-0.5897620622908205,-0.08241845654707416,3.226',
    n_fields=9)

print('yes')


# 完整演示一个例子：
# 读取文件名数据集，再将其合并得到新数据集
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shu
                       )
# https: // www.bilibili.com / video / av79196096?p = 47
# 4:21