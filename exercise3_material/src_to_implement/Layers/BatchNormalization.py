import numpy as np
from Layers import Base,Helpers
class BatchNormalization(Base.BaseLayer):
    def __init__(self,channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.input_tensor = None
        self.moving_motument = 0.8
        self.epsilon = np.finfo(np.float64).eps
        self.optimizer = None
        self.initialize()
    def initialize(self,weights_initializer_method=None, bias_initializer_method=None):
        self.weights = np.ones((1,self.channels))
        self.bias= np.zeros((1,self.channels))
        self.me_global = None
        self.va_global = None
        self.me_batch = 0
        self.va_batch = 1

    def reformat(self,tensor):
        if len(tensor.shape) == 2:#(2=>4)
            if len(self.input_tensor.shape)==2:
                return tensor
            else:
                bs, c,h, w = self.input_tensor.shape
                tmp =tensor.reshape(bs, h, w, c)
                return np.transpose(tmp,(0, 3, 1, 2))
        else:#(4=>2)
            if self.input_tensor is not None:
                if len(self.input_tensor.shape)==2:
                    return tensor
            c= tensor.shape[1]   
            data_transposed = np.transpose(tensor, (0, 2, 3, 1))  # (bs, h, w, c)
            tensor = data_transposed.reshape(-1, c)  # (bs*h*w, c)
            return tensor
    
    def forward(self,input_tensor):
        #(bs,c,w,h) / (dim,c)
        self.input_tensor = input_tensor
        input_tensor=self.reformat(input_tensor)
        if self.testing_phase is False:
            self.me_batch = np.mean(input_tensor,axis=(0), keepdims=True)
            self.va_batch = np.var(input_tensor,axis=(0), keepdims=True)
            if self.me_global is None:
                self.me_global = self.me_batch
                self.va_global = self.va_batch
            self.me_global = self.moving_motument*self.me_global + (1-self.moving_motument)*self.me_batch
            self.va_global = self.moving_motument*self.va_global + (1-self.moving_motument)*self.va_batch
        else:
            self.me_batch = self.me_global
            self.va_batch = self.va_global
        self.x = (input_tensor - self.me_batch) / np.sqrt(self.va_batch+self.epsilon)
        self.output = self.x*self.weights + self.bias
        return self.reformat(self.output)

    def backward(self,error_tensor):
        error_tensor = self.reformat(error_tensor)
        self.gradient_weights = np.sum(error_tensor * self.x, axis=0, keepdims=True)
        self.gradient_bias= np.sum(error_tensor, axis=0, keepdims=True)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias,self.gradient_bias)
        grad_input = Helpers.compute_bn_gradients(error_tensor,self.reformat(self.input_tensor),self.weights,self.me_batch,self.va_batch)
        return self.reformat(grad_input)
        
