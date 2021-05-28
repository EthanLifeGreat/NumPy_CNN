from Modules.nn import *
import time
import numpy as np
from utils.mnist import read_data_sets
from utils.one_hot import encode_one_hot
import sys

hidden_size = 512


class ConvNetwork(SequentialNeuralNetwork):
    def __init__(self, output_size):
        # NOTE: feel free to change structure and seed
        sequential = list()
        sequential.append(ConvolutionModule(1, 32, window_size=(5, 5), stride=(1, 1), padding=(2, 2)))
        sequential.append(ReluModule())
        sequential.append(MaxPoolModule(window_size=(2, 2), stride=(2, 2)))
        sequential.append(DropoutModule(p=0.15))
        sequential.append(ConvolutionModule(32, 64, window_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        sequential.append(ReluModule())
        sequential.append(MaxPoolModule(window_size=(2, 2), stride=(2, 2)))
        sequential.append(DropoutModule(p=0.15))
        
        sequential.append(LinearModule(7 * 7 * 64, hidden_size))
        sequential.append(ReluModule())
        sequential.append(LinearModule(hidden_size, hidden_size))
        sequential.append(ReluModule())
        sequential.append(DropoutModule(p=0.25))
        sequential.append(LinearModule(hidden_size, hidden_size))
        sequential.append(ReluModule())
        sequential.append(LinearModule(hidden_size, hidden_size))
        sequential.append(ReluModule())
        sequential.append(DropoutModule(p=0.25))
        sequential.append(LinearModule(hidden_size, output_size))

        loss_func = CrossEntropyLoss()
        optimizer = Adam(lr=1e-3)
        super().__init__(sequential, loss_func, optimizer)

batch_size = 100
display_size = 10000
shuffle = True
seed = 20210527

# Set Seed
np.random.seed(seed)

# read data
train_data, _, test_data = read_data_sets('./data/', val_size=0)
x_train, x_test = train_data[0] * 2 - 1, test_data[0] * 2 - 1
y_train, y_test = train_data[1], test_data[1]

# access data size: num_samples & num_features
num_samples, height, width, num_channels = x_train.shape
num_features = height * width
num_classes = 10

# Initialize
# model = LinearNetwork(num_features, num_classes)
model = ConvNetwork(num_classes)

# encode one-hot vector
y_01 = encode_one_hot(y_train)

start = time.time()
num_batches = num_samples // batch_sizes

# Training

num_epochs = 20
start = time.time()
num_batches = num_samples // batch_size
for epoch in range(num_epochs):
    loss = 0
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        x_train = x_train[shuffle_indices, :]
        y_01 = y_01[shuffle_indices, :]
    for batch in range(num_batches):
        batch_idc = range(batch * batch_size, (batch + 1) * batch_size)
        loss += model.train(x_train[batch_idc, :], y_01[batch_idc, :])
        if ((batch+1) * batch_size) % display_size == 0:
            print('Epoch [{}/{}], Sample [{}/{}], Loss: {:.5f}'
                  .format(epoch + 1, num_epochs, (batch+1) * batch_size, num_samples, loss/display_size))
            loss = 0


end = time.time()
print('Training Time:{} seconds'.format(end - start))