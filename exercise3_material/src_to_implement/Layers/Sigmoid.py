from Layers import Base
import numpy as np
class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self,input_tensor):
        self.sig = 1/(1+np.exp(-1* input_tensor))
        return self.sig
    def backward(self,error_tensor):
        return self.sig*(1-self.sig) * error_tensor