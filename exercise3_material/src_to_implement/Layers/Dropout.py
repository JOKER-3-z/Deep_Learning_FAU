import numpy as np
from Layers.Base import BaseLayer
class Dropout(BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.testing_phase = False
        self.probability=probability
        self.mask = None
    def forward(self,input_tensor):
        if self.testing_phase==False:
            self.mask =  (np.random.rand(*input_tensor.shape) < self.probability).astype(np.float32)
            return input_tensor * self.mask / self.probability
        else:
            return input_tensor
    def backward(self,error_tensor):
        return error_tensor * self.mask / self.probability