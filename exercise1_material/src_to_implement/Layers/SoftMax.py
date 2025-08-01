import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self,input_tensor):
        '''
            input_tensor: (bs,outsize)
            output: (bas,outsize)
        '''
        exp_tensor =  np.exp(input_tensor - np.max(input_tensor, axis=-1, keepdims=True)) #make sure element stability
        self.output = exp_tensor / np.sum(exp_tensor,axis=-1,keepdims=True)
        return self.output
    def backward(self,error_tensor):
        '''
            bs,outsize
            &x/&L= y*(e-sum(e*y))
        '''
        grad_tensor = self.output * (error_tensor - np.sum(error_tensor*self.output,axis=1, keepdims=True))
        return grad_tensor