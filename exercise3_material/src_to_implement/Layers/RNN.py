import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid  

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True
        self._memorize = False

        # 两个全连接层
        self.F1 = FullyConnected(hidden_size + input_size, hidden_size)  # 输入到隐藏
        self.F2 = FullyConnected(hidden_size, output_size)               # 隐藏到输出
        super().__init__(trainable=True)

        # 初始化
        self.gradient_weights_f1 = np.zeros_like(self.F1.weights)
        self.gradient_weights_f2 = np.zeros_like(self.F2.weights)
        self.tan_h = TanH()
        self.sigmoid = Sigmoid()
        self.h_t = None
        self.prev_h_t = None
        self.batch_size = None
        self.optimizer = None

        self.f1_inputs = []
        self.f2_inputs = []

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.input_tensor = input_tensor

        if self._memorize and self.h_t is not None:
            h0 = self.prev_h_t
        else:
            h0 = np.zeros((self.hidden_size,))

        self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))
        self.h_t[0] = h0

        self.f1_inputs.clear()
        self.f2_inputs.clear()

        y_t = np.zeros((self.batch_size, self.output_size))

        for t in range(self.batch_size):
            combined_input = np.hstack([self.h_t[t], input_tensor[t]])
            h_linear = self.F1.forward(combined_input[np.newaxis, :])
            self.f1_inputs.append(self.F1.input_tensor.copy())

            h_act = self.tan_h.forward(h_linear)
            self.h_t[t + 1] = h_act.squeeze()

            y_linear = self.F2.forward(h_act)
            self.f2_inputs.append(self.F2.input_tensor.copy())

            y_t[t] = y_linear.squeeze()

        self.prev_h_t = self.h_t[-1]
        return y_t

    def backward(self, error_tensor):
        self.gradient_weights_f1 = np.zeros_like(self.F1.weights)
        self.gradient_weights_f2 = np.zeros_like(self.F2.weights)

        hidden_error = np.zeros((self.batch_size + 1, self.hidden_size))
        out_error = np.zeros((self.batch_size, self.input_size))

        for t in reversed(range(self.batch_size)):
            self.F2.input_tensor = self.f2_inputs[t]
            dh_output = self.F2.backward(error_tensor[t][np.newaxis, :])
            self.gradient_weights_f2 += self.F2.gradient_weights

            total_dh = hidden_error[t + 1] + dh_output
            dh_raw = total_dh * (1 - self.h_t[t + 1] ** 2)

            self.F1.input_tensor = self.f1_inputs[t]
            dxh = self.F1.backward(dh_raw)
            self.gradient_weights_f1 += self.F1.gradient_weights

            hidden_error[t] = dxh[:, :self.hidden_size]
            out_error[t] = dxh[:, self.hidden_size:]

        self.out_error = out_error

        if self.optimizer is not None:
                self.F1.weights = self.optimizer.calculate_update(self.F1.weights, self.gradient_weights_f1)
                self.F2.weights = self.optimizer.calculate_update(self.F2.weights, self.gradient_weights_f2)
                self.weights = self.F1.weights

        return self.out_error

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.F2.initialize(weights_initializer, bias_initializer)
        self.F1.initialize(weights_initializer, bias_initializer)
        self.weights = self.F1.weights
        self.weights_y = self.F2.weights
        self.weights_h = self.F1.weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_f1

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.gradient_weights_f1 = value

    @property
    def weights(self):
        return self.F1.weights

    @weights.setter
    def weights(self, value):
        self.F1.weights = value