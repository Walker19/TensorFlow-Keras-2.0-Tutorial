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
# list or np.array ...
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

