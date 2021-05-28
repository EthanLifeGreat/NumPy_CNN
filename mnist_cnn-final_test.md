```python
from Modules.nn import *

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

```

    Using Torch CPU backend.



```python
import time
import numpy as np
from utils.mnist import read_data_sets
from utils.one_hot import encode_one_hot
import sys

batch_size = 100
display_size = 10000
shuffle = True
seed = 20210527

f0 = sys.stdout
f = open('./result/' + "batch_size{}_seed{}_console_rec.txt".format(batch_size, seed), 'w')
sys.stdout = f0

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
num_batches = num_samples // batch_size
    
```

    Extracting ./data/train-images-idx3-ubyte.gz
    Extracting ./data/train-labels-idx1-ubyte.gz
    Extracting ./data/t10k-images-idx3-ubyte.gz
    Extracting ./data/t10k-labels-idx1-ubyte.gz



```python
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

```

    Epoch [1/20], Sample [10000/60000], Loss: 1.61177
    Epoch [1/20], Sample [20000/60000], Loss: 0.43902
    Epoch [1/20], Sample [30000/60000], Loss: 0.19693
    Epoch [1/20], Sample [40000/60000], Loss: 0.15055
    Epoch [1/20], Sample [50000/60000], Loss: 0.12691
    Epoch [1/20], Sample [60000/60000], Loss: 0.11382
    Epoch [2/20], Sample [10000/60000], Loss: 0.08851
    Epoch [2/20], Sample [20000/60000], Loss: 0.09142
    Epoch [2/20], Sample [30000/60000], Loss: 0.08890
    Epoch [2/20], Sample [40000/60000], Loss: 0.08841
    Epoch [2/20], Sample [50000/60000], Loss: 0.08449
    Epoch [2/20], Sample [60000/60000], Loss: 0.07396
    Epoch [3/20], Sample [10000/60000], Loss: 0.06873
    Epoch [3/20], Sample [20000/60000], Loss: 0.07494
    Epoch [3/20], Sample [30000/60000], Loss: 0.06140
    Epoch [3/20], Sample [40000/60000], Loss: 0.06346
    Epoch [3/20], Sample [50000/60000], Loss: 0.05423
    Epoch [3/20], Sample [60000/60000], Loss: 0.06509
    Epoch [4/20], Sample [10000/60000], Loss: 0.04406
    Epoch [4/20], Sample [20000/60000], Loss: 0.05203
    Epoch [4/20], Sample [30000/60000], Loss: 0.04915
    Epoch [4/20], Sample [40000/60000], Loss: 0.06935
    Epoch [4/20], Sample [50000/60000], Loss: 0.06042
    Epoch [4/20], Sample [60000/60000], Loss: 0.05051
    Epoch [5/20], Sample [10000/60000], Loss: 0.04066
    Epoch [5/20], Sample [20000/60000], Loss: 0.04209
    Epoch [5/20], Sample [30000/60000], Loss: 0.04984
    Epoch [5/20], Sample [40000/60000], Loss: 0.04649
    Epoch [5/20], Sample [50000/60000], Loss: 0.04681
    Epoch [5/20], Sample [60000/60000], Loss: 0.05075
    Epoch [6/20], Sample [10000/60000], Loss: 0.03946
    Epoch [6/20], Sample [20000/60000], Loss: 0.04077
    Epoch [6/20], Sample [30000/60000], Loss: 0.04475
    Epoch [6/20], Sample [40000/60000], Loss: 0.04809
    Epoch [6/20], Sample [50000/60000], Loss: 0.03879
    Epoch [6/20], Sample [60000/60000], Loss: 0.04901
    Epoch [7/20], Sample [10000/60000], Loss: 0.03789
    Epoch [7/20], Sample [20000/60000], Loss: 0.03656
    Epoch [7/20], Sample [30000/60000], Loss: 0.02980
    Epoch [7/20], Sample [40000/60000], Loss: 0.03855
    Epoch [7/20], Sample [50000/60000], Loss: 0.03047
    Epoch [7/20], Sample [60000/60000], Loss: 0.03846
    Epoch [8/20], Sample [10000/60000], Loss: 0.03290
    Epoch [8/20], Sample [20000/60000], Loss: 0.02918
    Epoch [8/20], Sample [30000/60000], Loss: 0.03372
    Epoch [8/20], Sample [40000/60000], Loss: 0.03584
    Epoch [8/20], Sample [50000/60000], Loss: 0.03093
    Epoch [8/20], Sample [60000/60000], Loss: 0.04118
    Epoch [9/20], Sample [10000/60000], Loss: 0.02547
    Epoch [9/20], Sample [20000/60000], Loss: 0.02666
    Epoch [9/20], Sample [30000/60000], Loss: 0.02880
    Epoch [9/20], Sample [40000/60000], Loss: 0.03186
    Epoch [9/20], Sample [50000/60000], Loss: 0.03348
    Epoch [9/20], Sample [60000/60000], Loss: 0.03412
    Epoch [10/20], Sample [10000/60000], Loss: 0.02376
    Epoch [10/20], Sample [20000/60000], Loss: 0.02887
    Epoch [10/20], Sample [30000/60000], Loss: 0.02919
    Epoch [10/20], Sample [40000/60000], Loss: 0.02202
    Epoch [10/20], Sample [50000/60000], Loss: 0.04487
    Epoch [10/20], Sample [60000/60000], Loss: 0.03170
    Epoch [11/20], Sample [10000/60000], Loss: 0.02745
    Epoch [11/20], Sample [20000/60000], Loss: 0.02042
    Epoch [11/20], Sample [30000/60000], Loss: 0.02741
    Epoch [11/20], Sample [40000/60000], Loss: 0.02562
    Epoch [11/20], Sample [50000/60000], Loss: 0.02486
    Epoch [11/20], Sample [60000/60000], Loss: 0.02884
    Epoch [12/20], Sample [10000/60000], Loss: 0.02731
    Epoch [12/20], Sample [20000/60000], Loss: 0.02431
    Epoch [12/20], Sample [30000/60000], Loss: 0.02884
    Epoch [12/20], Sample [40000/60000], Loss: 0.02968
    Epoch [12/20], Sample [50000/60000], Loss: 0.02957
    Epoch [12/20], Sample [60000/60000], Loss: 0.02191
    Epoch [13/20], Sample [10000/60000], Loss: 0.01636
    Epoch [13/20], Sample [20000/60000], Loss: 0.02723
    Epoch [13/20], Sample [30000/60000], Loss: 0.01723
    Epoch [13/20], Sample [40000/60000], Loss: 0.02854
    Epoch [13/20], Sample [50000/60000], Loss: 0.02529
    Epoch [13/20], Sample [60000/60000], Loss: 0.02668
    Epoch [14/20], Sample [10000/60000], Loss: 0.01783
    Epoch [14/20], Sample [20000/60000], Loss: 0.01880
    Epoch [14/20], Sample [30000/60000], Loss: 0.01887
    Epoch [14/20], Sample [40000/60000], Loss: 0.02776
    Epoch [14/20], Sample [50000/60000], Loss: 0.02027
    Epoch [14/20], Sample [60000/60000], Loss: 0.02215
    Epoch [15/20], Sample [10000/60000], Loss: 0.02172
    Epoch [15/20], Sample [20000/60000], Loss: 0.02296
    Epoch [15/20], Sample [30000/60000], Loss: 0.02504
    Epoch [15/20], Sample [40000/60000], Loss: 0.01781
    Epoch [15/20], Sample [50000/60000], Loss: 0.01727
    Epoch [15/20], Sample [60000/60000], Loss: 0.02076
    Epoch [16/20], Sample [10000/60000], Loss: 0.01912
    Epoch [16/20], Sample [20000/60000], Loss: 0.02561
    Epoch [16/20], Sample [30000/60000], Loss: 0.01675
    Epoch [16/20], Sample [40000/60000], Loss: 0.01623
    Epoch [16/20], Sample [50000/60000], Loss: 0.03518
    Epoch [16/20], Sample [60000/60000], Loss: 0.02485
    Epoch [17/20], Sample [10000/60000], Loss: 0.01750
    Epoch [17/20], Sample [20000/60000], Loss: 0.02242
    Epoch [17/20], Sample [30000/60000], Loss: 0.01952
    Epoch [17/20], Sample [40000/60000], Loss: 0.01227
    Epoch [17/20], Sample [50000/60000], Loss: 0.02405
    Epoch [17/20], Sample [60000/60000], Loss: 0.01684
    Epoch [18/20], Sample [10000/60000], Loss: 0.02116
    Epoch [18/20], Sample [20000/60000], Loss: 0.01527
    Epoch [18/20], Sample [30000/60000], Loss: 0.01494
    Epoch [18/20], Sample [40000/60000], Loss: 0.01447
    Epoch [18/20], Sample [50000/60000], Loss: 0.01145
    Epoch [18/20], Sample [60000/60000], Loss: 0.02772
    Epoch [19/20], Sample [10000/60000], Loss: 0.02389
    Epoch [19/20], Sample [20000/60000], Loss: 0.01988
    Epoch [19/20], Sample [30000/60000], Loss: 0.01596
    Epoch [19/20], Sample [40000/60000], Loss: 0.01573
    Epoch [19/20], Sample [50000/60000], Loss: 0.01636
    Epoch [19/20], Sample [60000/60000], Loss: 0.01989
    Epoch [20/20], Sample [10000/60000], Loss: 0.01226
    Epoch [20/20], Sample [20000/60000], Loss: 0.01512
    Epoch [20/20], Sample [30000/60000], Loss: 0.01353
    Epoch [20/20], Sample [40000/60000], Loss: 0.01710
    Epoch [20/20], Sample [50000/60000], Loss: 0.02534
    Epoch [20/20], Sample [60000/60000], Loss: 0.02037
    Training Time:12995.835790157318 seconds



```python
# Testing

correct = 0
print('Testing...', end="")
num_test_samples = x_test.shape[0]
for batch in range(num_test_samples // batch_size):
    batch_idc = range(batch * batch_size, (batch + 1) * batch_size)
    y_hat = model.predict(x_test[batch_idc, :])
    correct += np.sum(y_hat == y_test[batch_idc])
    if ((batch + 1) * batch_size) % (num_test_samples // 10) == 0:
        print('.', end="")
print('\n{} Test Samples Accuracy:{}'.format(num_test_samples, correct / num_test_samples))

```

    Testing.............
    10000 Test Samples Accuracy:0.9933

