import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(trainable=True)

        self.weights_shape = (input_size, output_size)
        self.bias_shape = (1, output_size)

        # 默认初始化权重（即使不调用 initialize 也可工作）
        weights_part = np.random.uniform(0, 1, self.weights_shape)
        bias_part = np.random.uniform(0, 1, self.bias_shape)
        self.weights = np.vstack((weights_part, bias_part))

        self.gradient_weights = np.zeros((input_size + 1, output_size))
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, op):
        self._optimizer = op

    def initialize(self, weights_initializer, bias_initializer):
        weights_part = weights_initializer.initialize(
            self.weights_shape,
            self.weights_shape[0],
            self.weights_shape[1]
        )
        bias_part = bias_initializer.initialize(
            self.bias_shape,
            self.weights_shape[0],
            self.weights_shape[1]
        )
        self.weights = np.vstack((weights_part, bias_part))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.dot(input_tensor, self.weights[:-1, :]) + self.weights[-1:, :]

    def backward(self, error_tensor):
        self.gradient_weights[:-1, :] = np.dot(self.input_tensor.T, error_tensor)
        self.gradient_weights[-1:, :] = np.sum(error_tensor, axis=0)
        self.error_pre = np.dot(error_tensor, self.weights[:-1, :].T)

        if self._optimizer is not None:
            self.weights[:-1, :] = self._optimizer.calculate_update(
                self.weights[:-1, :], self.gradient_weights[:-1, :]
            )
            self.weights[-1:, :] = self._optimizer.calculate_update(
                self.weights[-1:, :], self.gradient_weights[-1:, :]
            )

        return self.error_pre
