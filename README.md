# NumPy_CNN

## （Chinese Guide)[NumPy_CNN：由NumPy实现的卷积神经网络中文介绍](https://github.com/EthanLifeGreat/NumPy_CNN/blob/main/README.CN.md)

# 中文版介绍

## Preview
NumPy_CNN contains CNN modules that is implemented in pure NumPy. Using these modules (defiend in Modules directory), in a PyTorch-like manner, we can easily build CNNs.

## Meaning
The meaning of this project is to, by converting optimized C codes into Python codes, demostrate how a CNN works (in my understanding).

## Demostration
In [mnist_cnn-final_test.md](https://github.com/EthanLifeGreat/NumPy_CNN/blob/main/mnist_cnn-final_test.md) one can see the format for building a CNN, which can achieve 99.33% test accuracy on MNIST.

## How to use
First download the whold directory.
1. If you have jupyter notebook, try to run main.ipynb.
2. Otherwise, run main.py, which does the same job.

Note that it may take a while to run with the default setting, so you may like to train with your prefered parameters.

## Introduction to the files
### main files
The two main files both first declared a CNN class, defining its network structure, the loss function and the optimizer. Then it starts training and testing.

With similar way of defining the network, you can build your own network, including CNN and simple BP.


### Modules 文件夹
nn.py: Similar to torch.nn in PyTorch, nn.py defines network layers(convolution, linear), loss functions(MSE, CrossEntropy) and optimizers(Adam, RMSProp) classes, which will be used in defining networks.

matrix_functions.py: is used to boost functions matmul() and einsum(), when you have installed torch. This is because NumPy often does not use multi-kernel to compute matrix multiply. However, without torch module, the program is still runnable.


### utils 文件夹
mnist.py is used to read MNIST dataset in directory data. The code is not written by us, and we want to thank the original author.

one_hot.py is used to encode one hot for labels.
