import numpy as np
import copy
class NeuralNetwork():
    def __init__(self,optimizer):
        self.optimizer=optimizer
        self.loss=[]
        self.layers=[]
        self.data_layer=None
        self.loss_layer=None
        self.label=None
        self.output=None
    def forward(self):
        input_tensor,label_tensor=self.data_layer.next()
        self.label=label_tensor
        output_tensor = input_tensor.copy()
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        self.output=self.loss_layer.forward(output_tensor,self.label)
        return self.output
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    def append_layer(self,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self,iterations):
        for it in range(iterations):
            self.loss.append(self.forward())
            self.backward()
    def test(self,input_tensor):
        output_tensor = input_tensor.copy()
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor


    
