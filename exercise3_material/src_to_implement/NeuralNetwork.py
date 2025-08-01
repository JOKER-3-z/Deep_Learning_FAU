import numpy as np
import copy
class NeuralNetwork():
    def __init__(self,optimizer,Initial_weights_method,Initial_bias_method):
        self.optimizer=optimizer
        self.Initial_weights_method=Initial_weights_method
        self.Initial_bias_method=Initial_bias_method
        self.loss=[]
        self.layers=[]
        self.data_layer=None
        self.loss_layer=None
        self.label=None
        self.output=None
        self.__phase = None
    @property
    def get_phase(self):
        return self.__phase
    @property
    def set_phase(self,phase):
        self.__phase=phase
    def forward(self):
        input_tensor,label_tensor=self.data_layer.next()
        self.label=label_tensor
        output_tensor = input_tensor.copy()
        norm_tensor=0
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
            try:
                norm_tensor += self.optimizer.regularizer.norm(layer.weights)
            except:
                pass
        self.output=self.loss_layer.forward(output_tensor,self.label)
        return self.output+norm_tensor
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    def append_layer(self,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.Initial_weights_method,self.Initial_bias_method)
        self.layers.append(layer)

    def train(self,iterations):
        for it in range(iterations):
            self.loss.append(self.forward())
            self.backward()
    def test(self,input_tensor):
        output_tensor = input_tensor.copy()
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = True
            output_tensor = layer.forward(output_tensor)
        return output_tensor


    
