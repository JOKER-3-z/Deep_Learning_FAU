from Layers import Base
import numpy as np
class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self,input_tensor):
        self.tn = np.tanh(input_tensor)
        return self.tn
    def backward(self,error_tensor):
        return (1 - self.tn**2)*error_tensor