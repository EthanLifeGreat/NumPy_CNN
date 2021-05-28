# 中文版介绍

## 简介 
NumPy_CNN 是仅使用NumPy包实现的CNN模块。使用这些（定义在Modules文件夹中的）模块，按照类似PyTorch的定义方式，我们可以轻松搭建卷积神经网络进行计算。

## 意义
本项目的意义在于展示出简明的Python代码，最终展示出CNN的原理。（在我理解中的原理）。

## 展示模块
打开 mnist_cnn-final_test.md 可以看到调用这些模块的代码格式和运行结果，可以达到99.33%的测试集精度。

## 使用方法
首先下载整个文件夹。
1. 如果你有jupyter notebook，那么可以试着运行main.ipynb。
2. 否则，可以运行main.py，其结果与1是一样的。

注意，当前默认的参数可能会需要较长的时间，可以修改训练参数来进行更简短的模型训练。

## 文件介绍
### main文件
两个main文件都首先声明了一个CNN类，并定义了它的网络结构，损失函数和优化器。然后进行训练和测试。
按照类似的定义方式，你可以声明自己的神经网络，包括CNN或者简单的BP网络。

### Modules 文件夹
nn.py 仿照 PyTorch.nn 类，定义了不同的神经网络层（卷积层、线性层），损失函数（MSE、Cross Entropy）和优化器（Adam、RMSProp）的类。在定义网络结构的时候需要用到它们。
matrix_functions.py 是对矩阵函数matmul和einsum进行加速的，在你安装了torch包时，它将极大地加速计算。这是因为通常NumPy不支持CPU的矩阵并行计算，而torch则支持。而即使没有安装torch包，程序仍然可以正确运行。

### utils 文件夹
mnist.py 用于对data文件夹中的MNIST数据集进行读取。并非本人所写，感谢原作者的贡献。
one_hot.py 用于对标签进行one hot编码。
