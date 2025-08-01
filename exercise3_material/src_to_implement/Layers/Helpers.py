import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import struct
import random
from pathlib import Path
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, load_digits


def gradient_check(layers, input_tensor, label_tensor, seed = None):
    epsilon = 1e-5
    difference = np.zeros_like(input_tensor)

    activation_tensor = input_tensor.copy()
    for layer in layers[:-1]:
        np.random.seed(seed) if seed is not None else None
        random.seed(seed) if seed is not None else None
        activation_tensor = layer.forward(activation_tensor)
    layers[-1].forward(activation_tensor, label_tensor)

    error_tensor = layers[-1].backward(label_tensor)
    for layer in reversed(layers[:-1]):
        error_tensor = layer.backward(error_tensor)

    it = np.nditer(input_tensor, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = input_tensor.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = input_tensor.copy()
        minus_epsilon[it.multi_index] -= epsilon

        analytical_derivative = error_tensor[it.multi_index]

        for layer in layers[:-1]:
            np.random.seed(seed) if seed is not None else None
            random.seed(seed) if seed is not None else None
            plus_epsilon = layer.forward(plus_epsilon)
            np.random.seed(seed) if seed is not None else None
            random.seed(seed) if seed is not None else None
            minus_epsilon = layer.forward(minus_epsilon)
        upper_error = layers[-1].forward(plus_epsilon, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

        # print('Analytical: ' + str(analytical_derivative) + ' vs Numerical :' + str(numerical_derivative))
        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant
        it.iternext()
    return difference


def gradient_check_weights(layers, input_tensor, label_tensor, bias):
    epsilon = 1e-5
    if bias:
        weights = layers[0].bias
    else:
        weights = layers[0].weights
    difference = np.zeros_like(weights)

    it = np.nditer(weights, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = weights.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = weights.copy()
        minus_epsilon[it.multi_index] -= epsilon

        activation_tensor = input_tensor.copy()
        if bias:
            layers[0].bias = weights
        else:
            layers[0].weights = weights
        for layer in layers[:-1]:
            activation_tensor = layer.forward(activation_tensor)
        layers[-1].forward(activation_tensor, label_tensor)

        error_tensor = layers[-1].backward(label_tensor)
        for layer in reversed(layers[:-1]):
            error_tensor = layer.backward(error_tensor)
        if bias:
            analytical_derivative = layers[0].gradient_bias
        else:
            analytical_derivative = layers[0].gradient_weights
        analytical_derivative = analytical_derivative[it.multi_index]

        if bias:
            layers[0].bias = plus_epsilon
        else:
            layers[0].weights = plus_epsilon
        plus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            plus_epsilon_activation = layer.forward(plus_epsilon_activation)

        if bias:
            layers[0].bias = minus_epsilon
        else:
            layers[0].weights = minus_epsilon
        minus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            minus_epsilon_activation = layer.forward(minus_epsilon_activation)

        upper_error = layers[-1].forward(plus_epsilon_activation, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon_activation, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant

        it.iternext()
    return difference


def compute_bn_gradients(error_tensor, input_tensor, weights, mean, var, eps=np.finfo(float).eps):
    # computation of the gradient w.r.t the input for the batch_normalization layer

    if eps > 1e-10:
        raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" %(str(eps)))

    norm_mean = input_tensor - mean
    var_eps = var + eps

    gamma_err = error_tensor * weights
    inv_batch = 1. / error_tensor.shape[0]

    grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

    sqrt_var = np.sqrt(var_eps)
    first = gamma_err * 1. / sqrt_var

    grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=0)) * inv_batch
    grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=0)

    second = grad_var * (2. * norm_mean) * inv_batch
    grad_mu = grad_mu_two + grad_mu_one

    return first + second + inv_batch * grad_mu


def calculate_accuracy(results, labels):

    index_maximum = np.argmax(results, axis=1)
    one_hot_vector = np.zeros_like(results)
    for i in range(one_hot_vector.shape[0]):
        one_hot_vector[i, index_maximum[i]] = 1

    correct = 0.
    wrong = 0.
    for column_results, column_labels in zip(one_hot_vector, labels):
        if column_results[column_labels > 0.].all() > 0.:
            correct += 1.
        else:
            wrong += 1.

    return correct / (correct + wrong)


def shuffle_data(input_tensor, label_tensor):
    index_shuffling = [i for i in range(input_tensor.shape[0])]
    shuffle(index_shuffling)
    shuffled_input = [input_tensor[i, :] for i in index_shuffling]
    shuffled_labels = [label_tensor[i, :] for i in index_shuffling]
    return (np.array(shuffled_input)), (np.array(shuffled_labels))


class RandomData:
    def __init__(self, input_size, batch_size, categories):
        self.input_size = input_size
        self.batch_size = batch_size
        self.categories = categories
        self.label_tensor = np.zeros([self.batch_size, self.categories])

    def next(self):
        input_tensor = np.random.random([self.batch_size, self.input_size])

        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        return input_tensor, self.label_tensor


class IrisData:
    def __init__(self, batch_size, random=True):
        self.random = random
        self.batch_size = batch_size
        self._data = load_iris()
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(self._data.target.reshape(-1, 1))
        self._input_tensor = self._data.data
        self._input_tensor /= np.abs(self._input_tensor).max()

        self.split = int(self._input_tensor.shape[0]*(2/3))  # train / test split  == number of samples in train set

        self._input_tensor, self._label_tensor = shuffle_data(self._input_tensor, self._label_tensor)
        self._input_tensor_train = self._input_tensor[:self.split, :]
        self._label_tensor_train = self._label_tensor[:self.split, :]
        self._input_tensor_test = self._input_tensor[self.split:, :]
        self._label_tensor_test = self._label_tensor[self.split:, :]

        self._current_forward_idx_iterator = self._forward_idx_iterator()

    def _forward_idx_iterator(self):
        num_iterations = int(np.ceil(self.split / self.batch_size))
        idx = np.arange(self.split)
        while True:
            this_idx = np.random.choice(idx, self.split, replace=False) if self.random else idx
            for i in range(num_iterations):
                yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]

    def next(self):
        idx = next(self._current_forward_idx_iterator)
        return self._input_tensor_train[idx, :], self._label_tensor_train[idx, :]

    def get_test_set(self):
        return self._input_tensor_test, self._label_tensor_test

class DigitData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._data = load_digits(n_class=10)
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(self._data.target.reshape(-1, 1))
        self._input_tensor = self._data.data.reshape(-1, 1, 8, 8)
        self._input_tensor /= np.abs(self._input_tensor).max()

        self.split = int(self._input_tensor.shape[0]*(2/3))  # train / test split  == number of samples in train set

        self._input_tensor, self._label_tensor = shuffle_data(self._input_tensor, self._label_tensor)
        self._input_tensor_train = self._input_tensor[:self.split, :]
        self._label_tensor_train = self._label_tensor[:self.split, :]
        self._input_tensor_test = self._input_tensor[self.split:, :]
        self._label_tensor_test = self._label_tensor[self.split:, :]

        self._current_forward_idx_iterator = self._forward_idx_iterator()

    def _forward_idx_iterator(self):
        num_iterations = int(np.ceil(self.split / self.batch_size))
        rest = self.batch_size-self.split%self.batch_size
        idx = np.arange(self.split)
        while True:
            this_idx = np.random.choice(idx, self.split, replace=False)
            for i in range(num_iterations):
                if (i == num_iterations-1) and (rest != 0):
                    yield np.concatenate([this_idx[i * self.batch_size:(i + 1) * self.batch_size], this_idx[:rest]])
                else:
                    yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]

    def next(self):
        idx = next(self._current_forward_idx_iterator)

        return self._input_tensor_train[idx, :], self._label_tensor_train[idx, :]

    def get_test_set(self):
        return self._input_tensor_test, self._label_tensor_test


class MNISTData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train, self.labels = self._read()
        self.test, self.testLabels = self._read(dataset="testing")

        self._current_forward_idx_iterator = self._forward_idx_iterator()

    def _forward_idx_iterator(self):
        num_iterations = int(self.train.shape[0] / self.batch_size)
        idx = np.arange(self.train.shape[0])
        while True:
            this_idx = np.random.choice(idx, self.train.shape[0], replace=False)
            for i in range(num_iterations):
                yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]

    def next(self):
        idx = next(self._current_forward_idx_iterator)
        return self.train[idx, :], self.labels[idx, :]

    def show_random_training_image(self):
        image = self.train[np.random.randint(0, self.train.shape[0]-1), :28 , :28]
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

    def show_image(self, index, test=True):
        if test:
            image = self.test[index, :28 * 28]
        else:
            image = self.train[index, :28 * 28]

        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

    def get_test_set(self):
        return self.test, self.testLabels

    @staticmethod
    def _read(dataset="training"):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """

        root_dir = Path(__file__)

        if dataset == "training":
            fname_img = root_dir.parent.parent.joinpath('Data', 'train-images-idx3-ubyte.gz')
            fname_lbl = root_dir.parent.parent.joinpath('Data', 'train-labels-idx1-ubyte.gz')
        elif dataset == "testing":
            fname_img = root_dir.parent.parent.joinpath('Data', 't10k-images-idx3-ubyte.gz')
            fname_lbl = root_dir.parent.parent.joinpath('Data', 't10k-labels-idx1-ubyte.gz')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Load everything in some numpy arrays
        with gzip.open(str(fname_lbl), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))

            s = flbl.read(num)
            lbl = np.frombuffer(s, dtype=np.int8)
            one_hot = np.zeros((lbl.shape[0],10))
            for idx, l in enumerate(lbl):
                one_hot[idx, l] = 1

        with gzip.open(str(fname_img), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))

            buffer = fimg.read(num * 32 * 32 * 8)
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(len(lbl), 1, rows,  cols)
            img = img.astype(np.float64)
            img /= 255.0

        img = img[:num, :]
        one_hot = one_hot[:num, :]
        return img, one_hot
