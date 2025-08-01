import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True
        self._memorize = False

        # FullyConnected 层
        self.FC_h = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_y = FullyConnected(hidden_size, output_size)
        super().__init__(trainable=True)

        # 初始化其余属性
        self.gradient_weights_h = np.zeros_like(self.FC_h.weights)
        self.gradient_weights_y = np.zeros_like(self.FC_y.weights)
        self.tan_h = TanH()
        self.h_t = None
        self.prev_h_t = None
        self.batch_size = None
        self.optimizer = None
        self.h_mem = []

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.input_tensor = input_tensor
        
        # 初始化隐藏状态
        if self._memorize and self.h_t is not None:
            self.h_t = np.vstack([self.prev_h_t, np.zeros((self.batch_size, self.hidden_size))])
        else:
            self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))
        
        y_t = np.zeros((self.batch_size, self.output_size))
        self.h_mem = []  # 保存每个时间步的输入组合

        for b in range(self.batch_size):
            # 组合隐藏状态和输入
            input_new = np.hstack([self.h_t[b], input_tensor[b]])
            self.h_mem.append(input_new)
            
            # 计算新隐藏状态
            w_t = self.FC_h.forward(input_new[np.newaxis, :])
            self.h_t[b+1] = self.tan_h.forward(w_t)
            
            # 计算输出
            y_t[b] = self.FC_y.forward(self.h_t[b+1][np.newaxis, :])
        
        self.prev_h_t = self.h_t[-1]
        return y_t

    def backward(self, error_tensor):
        self.out_error = np.zeros((self.batch_size, self.input_size))
        hidden_error = np.zeros((self.batch_size + 1, self.hidden_size))

        self.gradient_weights_y = np.zeros_like(self.FC_y.weights)
        self.gradient_weights_h = np.zeros_like(self.FC_h.weights)

        for b in reversed(range(self.batch_size)):
            # Output layer backward
            yh_error = self.FC_y.backward(error_tensor[b][np.newaxis, :])
            self.gradient_weights_y += self.FC_y.gradient_weights

            grad_hidden_total = hidden_error[b + 1] + yh_error
            grad_hidden = (1 - self.h_t[b + 1] ** 2) * grad_hidden_total

            xh_error = self.FC_h.backward(grad_hidden)
            self.gradient_weights_h += self.FC_h.gradient_weights

            hidden_error[b] = xh_error[:, :self.hidden_size]
            self.out_error[b] = xh_error[:, self.hidden_size:]

        if self.optimizer is not None:
            self.FC_y.weights = self.optimizer.calculate_update(self.FC_y.weights, self.gradient_weights_y)
            self.FC_h.weights = self.optimizer.calculate_update(self.FC_h.weights, self.gradient_weights_h)
            self.weights = self.FC_h.weights

        return self.out_error



    # 其余属性方法保持不变...
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
        self.FC_y.initialize(weights_initializer, bias_initializer)
        self.FC_h.initialize(weights_initializer, bias_initializer)
        self.weights = self.FC_h.weights
        self.weights_y = self.FC_y.weights
        self.weights_h = self.FC_h.weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_h

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.gradient_weights_h = value

    @property
    def weights(self):
        return self.FC_h.weights

    @weights.setter
    def weights(self, value):
        self.FC_h.weights = value
