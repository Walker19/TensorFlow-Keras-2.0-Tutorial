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
# continue
# https://www.bilibili.com/video/av79196096?p=46
