# TensorFlow-Keras-2.0-Tutorial
Tensorflow 2.0 教程，主要讲述tf.keras api

## Tensorflow2.0 入门到进阶
- 教程来源：https://www.bilibili.com/video/av79196096?p=1

## 第2章 建模

- 1. 重点学习tf.keras 2.0 api，模型搭建方法共三种：

  - 1. 创建Sequential类，实例化对象后拥有add方法，可以随意添加网络层

  ```python
  model = keras.models.Sequential()
  model.add(keras.layers.Flatten(input_shape=[28, 28]))  # 模型的第一层，功能是将28*28的输入矩阵拉平为向量
  model.add(keras.layers.Dense(300, activation="sigmoid"))
  model.add(keras.layers.Dense(100, activation="sigmoid"))
  model.add(keras.layers.Dense(10, activation="softmax"))
  ```

  - 2. 函数式api，直接实例化keras内置网络层，并通过call方法调用

  ```python
  inputs = keras.layers.Input(shape=x_train.shape[1:])  # 对了，注意这里只用输入每个样本的输入维度，无须输入batch维度
  hidden1 = keras.layers.Dense(30, activation="relu")(inputs)  # 这就是函数式API的特性，直接call即可
  hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
  # 函数式API效果类似复合函数：f(x) = h(g(x))
  
  concat = keras.layers.concatenate([inputs, hidden2])
  output = keras.layers.Dense(1)(concat)  # 回归问题，所以无须激活函数
  
  model = keras.models.Model(inputs=[inputs],
                             outputs=[output])
  ```

  **直接调用网络层的call方法即可，但是为了得到keras封装好的Model、需要调用keras.models.Model，并指定输入和输出，才得到完整的模型。**

  - 3. 子类api，自定义类，该类继承keras.models.Model或者keras.layers.Layer

  ```python
  class WideDeepModel(keras.models.Model):
      def __init__(self):
          super(WideDeepModel, self).__init__()
          """定义模型的层次"""
          self.hidden1_layer = keras.layers.Dense(30, activation="relu")
          self.hidden2_layer = keras.layers.Dense(30, activation="relu")
          self.output_layer = keras.layers.Dense(1)
  
      def call(self, input):
          """完成模型的前向计算"""
          hidden1 = self.hidden1_layer(input)
          hidden2 = self.hidden2_layer(hidden1)
          concat = keras.layers.concatenate([input, hidden2])
          output = self.output_layer(concat)
          return output
  
  # 1 方法一，实例化
  model = WideDeepModel()
  model.build(input_shape=(None, 8))  # build主要是为了告诉模型第一层输入的shape是怎样的
  ```

  或者继承keras.layers.Layer

  ```python
  class CustomizedDenseLayer(keras.layers.Layer):
      def __init__(self, units, activation=None, **kwargs):
          self.units = units
          self.activation = keras.layers.Activation(activation)
          super(CustomizedDenseLayer, self).__init__(**kwargs)  # 为什么？
  
      def build(self, input_shape):
          """构建所需要的参数"""
          # x * w + b;input_shape:[None, a]; w:[a, b]; output_shape:[None, b]
          self.kernel = self.add_weight(name='kernel',
                                        shape=(input_shape[1], self.units),  # 指定参数w的形状
                                        initializer='uniform',  # 参数使用均匀分布初始化
                                        trainable=True)
          self.bias = self.add_weight(name="bias",
                                      shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)
          super(CustomizedDenseLayer, self).build(input_shape)  # 为什么？
  
      def call(self, x):
          """完成正向计算"""
          return self.activation(x @ self.kernel + self.bias)
      
  model = keras.models.Sequential([
      CustomizedDenseLayer(30, activation='relu', input_shape=x_train.shape[1:]),
      CustomizedDenseLayer(1),
  ])
  model.summary()
  model.compile(loss="mean_squared_error", optimizer="sgd",  # 使用自定义损失函数
                metrics=["mean_squared_error"])
  ```


- 2. 再讲一下函数式api的调用原理：

     其中`Input`生成一个`symbolic tensor` (也即1.x中的占位符概念)，这个占位符可以作为TensorFlow中其他算子的输入。比如：

```python
x = Input(shape=(32,))
y = tf.square(x)
```

