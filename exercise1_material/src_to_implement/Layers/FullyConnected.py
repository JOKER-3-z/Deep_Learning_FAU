import numpy as np
from Layers.Base import BaseLayer
class FullyConnected(BaseLayer):
    def __init__(self,input_size,output_size):
        super().__init__(True,np.random.uniform(0,1,(input_size+1,output_size))) #weights + bias
        self.gradient_weights = np.zeros((input_size+1,output_size))
        self._optimizer = None
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self,op):
        self._optimizer=op
    def forward(self,input_tensor):
        '''
            input shape : (bs,input_size) 
            weights shape :(input_size , output_size)
            bias shape:(1,output_size)
            input * weight + bias = (bs,input_size)*(input_size,output_size)+(1,output_size)
            return (bs,output_size)
        '''
        self.input_tensor = input_tensor
        return np.dot(input_tensor,self.weights[:-1,:])+self.weights[-1:,:]
    def backward(self,error_tensor):
        '''
            &L/&W = self.input_tensor.T(inputsize,bs) * erro_tensor(bs,outsize) =(inputsize,outsize)
            &L/&b = error_tensor.sum to (1,outsize)
            error_pre = error_tensor(bs,output_size) * weight[:-1,:].T(output_size,input_size)
        '''
        self.gradient_weights[:-1,:] = np.dot(self.input_tensor.T , error_tensor)
        self.gradient_weights[-1:,:] = np.sum(error_tensor ,axis=0)
        self.error_pre = np.dot(error_tensor,self.weights[:-1,:].T)
        if self._optimizer != None:
            self.weights[:-1,:] = self._optimizer.calculate_update(self.weights[:-1,:],self.gradient_weights[:-1,:])
            self.weights[-1:,:] = self._optimizer.calculate_update(self.weights[-1:,:],self.gradient_weights[-1:,:])
        return self.error_pre

