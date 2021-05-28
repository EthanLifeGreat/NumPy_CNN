import numpy as np
from Modules.matrix_functions import get_matrix_functions
import copy

# Get boost from Torch's multi-kernel-computation if available
matmul, einsum = get_matrix_functions(boost=True, device='cpu')


def batch_image_unroll(img, weight_size, stride):
    # img is a 4d tensor([batch_size, height, width, channels])
    w_height, w_width = weight_size[0], weight_size[1]
    s_height, s_width = stride[0], stride[1]
    num_samples, x_height, x_width, num_channels = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    y_height, y_width = (x_height - w_height) // s_height + 1, (x_width - w_width) // s_width + 1
    unrolled_img = np.zeros([num_samples, y_height, y_width, w_height, w_width, num_channels])
    for y in range(w_height):
        y_max = y + s_height * y_height
        for x in range(w_width):
            x_max = x + s_width * y_width
            unrolled_img[:, :, :, y, x, :] = img[:, y:y_max:s_height, x:x_max:s_width, :]
    unrolled_img = unrolled_img.reshape([num_samples, y_height * y_width, w_height * w_width * num_channels])
    return unrolled_img, y_height, y_width


def batch_image_roll(unrolled_img, zero_image_shape, weight_size, stride, y_height, y_width, num_channels):
    # unrolled_x is a 3d tensor([batch_size, y_height * y_width, w_height * w_width * channels])
    num_samples = unrolled_img.shape[0]
    w_height, w_width = weight_size[0], weight_size[1]
    s_height, s_width = stride[0], stride[1]
    unrolled_img = unrolled_img.reshape([num_samples, y_height, y_width, w_height, w_width, num_channels])
    img = np.zeros(zero_image_shape)
    for y in range(w_height):
        y_max = y + s_height * y_height
        for x in range(w_width):
            x_max = x + s_width * y_width
            img[:, y:y_max:s_height, x:x_max:s_width, :] += unrolled_img[:, :, :, y, x, :]
    return img


def softmax(z):
    num_classes = z.shape[1]
    zk = np.expand_dims(z, 2).repeat(num_classes, axis=2)
    zi = np.expand_dims(z, 1).repeat(num_classes, axis=1)
    ret = zi - zk
    ret = 1 / np.sum(np.exp(ret), axis=2)
    return ret


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0
        self.v = 0
        self.t = 0

    def step(self, dw):
        self.t += 1
        g = dw
        m = (self.beta1 * self.m + (1 - self.beta1) * g)
        v = (self.beta2 * self.v + (1 - self.beta2) * g ** 2)
        alpha = self.alpha * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        dw = alpha * m / (np.sqrt(v) + self.eps)
        self.m = m
        self.v = v
        return dw


class RMSProp:
    def __init__(self, lr=1e-3, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.r = 0

    def step(self, dw):
        self.r = self.r * self.rho + (1-self.rho) * (dw ** 2)
        return self.lr / np.sqrt((self.eps + self.r)) * dw


class Sgd:
    def __init__(self, lr=1e-1):
        self.lr = lr

    def step(self, dw):
        return self.lr*dw


class NeuralNetworkModule:
    def forward(self, x):
        pass

    def backward(self, dy):
        pass

    def predict(self, x):
        return self.forward(x)


class OptimizableModule(NeuralNetworkModule):
    # Should fill w and dw in subclasses
    def __init__(self):
        self.optimizer = None
        self.w = None
        self.dw = None

    def update(self):
        step = self.optimizer.step(self.dw)
        self.w -= step


class SigmoidModule(NeuralNetworkModule):
    # Sigmoid Layer: y = 1 / (1 + e^(-x))
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        dx = dy * self.y * (1 - self.y)
        return dx


class ReluModule(NeuralNetworkModule):
    # ReLU Layer: y = 0, x<0; x, x>=0
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        y = x.copy()
        y[x < 0] = 0
        return y

    def backward(self, dy):
        d = np.ones_like(self.x)
        d[self.x < 0] = 0
        return d * dy


class DropoutModule(NeuralNetworkModule):
    def __init__(self, p):
        self.keep_prob = 1-p
        self.mask = None

    def forward(self, x):
        self.mask = np.random.binomial(1, self.keep_prob, size=x.shape)
        y = x * self.mask / self.keep_prob
        return y

    def backward(self, dy):
        dx = dy * self.mask * self.keep_prob
        return dx

    def predict(self, x):
        return x


class LinearModule(OptimizableModule):
    # Linear Layer: W * [X, 1] = Y
    def __init__(self, input_size, output_size, random=True):
        super().__init__()
        if random:
            std_v = 1. / np.sqrt(input_size)
            self.w = (np.random.uniform(-std_v, std_v, [input_size + 1, output_size]))
        else:
            self.w = np.zeros([input_size + 1, output_size])
        self.dw = None
        self.x_1 = None
        self.input_shape = None
        self.input_size = input_size

    def forward(self, x):
        num_samples = x.shape[0]
        self.input_shape = x.shape
        if len(x.shape) > 2:
            # Squeezing
            x = x.reshape(num_samples, self.input_size)
        self.x_1 = np.concatenate([x, np.ones([num_samples, 1])], axis=1)
        y = matmul(self.x_1, self.w)
        return y

    def backward(self, dy):
        self.dw = einsum('ij,ih->hj', dy, self.x_1) / dy.shape[0]
        dx = matmul(dy, np.transpose(self.w))[:, :-1]
        dx = dx.reshape(self.input_shape)
        return dx


class ConvolutionModule(OptimizableModule):
    # !!!!! should satisfy: 2p + x - w == (y - 1) * s !!!!!
    def __init__(self, num_channels_in, num_channels_out, window_size, padding, stride=(1, 1), random=True):
        # weight_size: [height, width]
        super().__init__()
        # unrolled weight
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.y_width = 0
        self.y_height = 0
        self.zero_image_shape = None
        self.weight_size = (window_size[0] * window_size[1] * (num_channels_in + 1), num_channels_out)
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        if random:
            std_v = 1. / np.sqrt(np.prod(self.weight_size) * num_channels_in)
            self.w = np.random.uniform(-std_v, std_v, self.weight_size)
        else:
            self.w = np.zeros(self.weight_size)
        self.dw = np.zeros_like(self.w)
        self.x1p = None

    def batch_unroll(self, x, weight_size, stride):
        # x is a 4d tensor([batch_size, height, width, channels])
        unrolled_x, y_height, y_width = batch_image_unroll(x, weight_size, stride)
        self.zero_image_shape = x.shape
        self.y_width, self.y_height = y_width, y_height
        return unrolled_x

    def batch_roll(self, unrolled_x, weight_size, stride):
        # unrolled_x is a 3d tensor([batch_size, y_height * y_width, w_height * w_width * channels])
        y_width, y_height = self.y_width, self.y_height
        num_channels = self.num_channels_in + 1
        x = batch_image_roll(unrolled_x, self.zero_image_shape, weight_size, stride, y_height, y_width, num_channels)
        return x

    @staticmethod
    def batch_matmul(x, w):
        return einsum('ijk,kh->ijh', x, w)

    def batch_convolve(self, x, w):
        # x is bias-included, batched, unrolled input
        # w is unrolled kernel
        y = self.batch_matmul(x, w)
        return y

    def forward(self, x):
        # x_size = [batch_size, height, width, num_channels_in]
        assert len(x.shape) == 4
        num_samples, x_height, x_width = x.shape[0], x.shape[1], x.shape[2]
        x_1 = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
        x_1_p = np.pad(x_1, ((0, 0), (self.padding[0], self.padding[0]),
                             (self.padding[1], self.padding[1]), (0, 0)), 'constant', constant_values=0)
        unrolled_x1p = self.batch_unroll(x_1_p, self.window_size, self.stride)
        self.x1p = unrolled_x1p
        unrolled_y = self.batch_convolve(self.x1p, self.w)
        y = unrolled_y.reshape([num_samples, self.y_height, self.y_width, self.num_channels_out])
        return y

    def backward(self, dy):
        num_samples, y_height, y_width = dy.shape[0], dy.shape[1], dy.shape[2]
        dy_unrolled = dy.reshape([num_samples, y_height * y_width, self.num_channels_out])
        dw = einsum('ijk,ikh->jh', np.transpose(self.x1p, axes=(0, 2, 1)), dy_unrolled)
        self.dw = dw / num_samples
        dx1p_unrolled = self.batch_matmul(dy_unrolled, np.transpose(self.w))
        dx1p = self.batch_roll(dx1p_unrolled, self.window_size, self.stride)
        dx1 = dx1p[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], :]
        return dx1[:, :, :, :-1]


class MaxPoolModule(NeuralNetworkModule):
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride
        self.mask = None
        self.y_height = 0
        self.y_width = 0
        self.output_size = 0
        self.zero_image_shape = None

    def forward(self, x):
        num_samples, num_channels = x.shape[0], x.shape[3]
        unrolled_x, y_height, y_width = batch_image_unroll(x, self.window_size, self.stride)
        self.zero_image_shape = x.shape
        self.output_size = num_samples * y_height * y_width * num_channels
        unrolled_x = unrolled_x.reshape([num_samples, y_height * y_width, np.prod(self.window_size), num_channels])
        unrolled_x_ = np.transpose(unrolled_x, axes=(0, 1, 3, 2)). \
            reshape(self.output_size, np.prod(self.window_size))
        y = np.max(unrolled_x, axis=2).reshape([num_samples, y_height, y_width, num_channels])
        self.mask = np.argmax(unrolled_x_, axis=1)
        self.y_height, self.y_width = y_height, y_width
        return y

    def backward(self, dy):
        # dy is a batched gradients with size [num_samples, y_height, y_width, num_channels]
        num_samples, y_height, y_width, num_channels = dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]
        dy_unrolled = dy.reshape([num_samples, y_height * y_width, 1, num_channels])
        dx_unrolled_ = np.zeros([self.output_size, np.prod(self.window_size)])
        dx_unrolled_[np.arange(self.output_size), self.mask] = dy_unrolled.ravel()
        dx_unrolled = np.transpose(dx_unrolled_.reshape(
            [num_samples, y_height * y_width, num_channels, np.prod(self.window_size)]), axes=(0, 1, 3, 2))
        dx_unrolled = dx_unrolled.reshape([num_samples, y_height * y_width, np.prod(self.window_size) * num_channels])
        dx = batch_image_roll(dx_unrolled, self.zero_image_shape,
                              self.window_size, self.stride, y_height, y_width, num_channels)
        return dx


class MeanSquaredError:
    def __call__(self, y_hat, y):
        # return Loss & Derivative
        return np.mean((y - y_hat) ** 2) / 2, y_hat - y


class CrossEntropyLoss:
    def __call__(self, y_hat, y):
        y_prob = softmax(y_hat)
        # return Loss & Derivative
        return np.sum(- np.log(y_prob) * y), y_prob - y


class SequentialNeuralNetwork:
    def __init__(self, sequential, loss_func, optimizer):
        self.sequential = sequential
        self.loss_func = loss_func
        self.set_optimizer(optimizer)

    def forward(self, x):
        h = x
        for layer in self.sequential:
            h = layer.forward(h)
        return h

    def backward(self, dy):
        dh = dy
        for layer in reversed(self.sequential):
            dh = layer.backward(dh)

    def update(self):
        for layer in self.sequential:
            if isinstance(layer, OptimizableModule):
                layer.update()

    def set_optimizer(self, optimizer):
        for layer in self.sequential:
            if isinstance(layer, OptimizableModule):
                layer.optimizer = copy.copy(optimizer)

    def train(self, x, y):
        y_hat = self.forward(x)
        (loss, dy) = self.loss_func(y_hat, y)
        self.backward(dy)
        self.update()
        return loss

    def predict(self, x):
        h = x
        for layer in self.sequential:
            h = layer.predict(h)
        y_hat = h
        return np.argmax(y_hat, 1)
