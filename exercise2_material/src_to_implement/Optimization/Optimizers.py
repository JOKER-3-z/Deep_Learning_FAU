import numpy as np

class Sgd():
    def __init__(self,lr):
        self.lr = lr
    def calculate_update(self,weight_tensor,gradient_tensor):
        return weight_tensor - self.lr * gradient_tensor
    
class  SgdWithMomentum:
    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v=None
    def calculate_update(self,weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        if isinstance(weight_tensor, np.ndarray):
            if self.v.shape != weight_tensor.shape:
                self.v = np.zeros_like(weight_tensor)
        self.v = self.momentum_rate*self.v - self.learning_rate * gradient_tensor
        weight_tensor = weight_tensor + self.v
        return weight_tensor

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None 
        self.t = 0    

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is  None:
            self.v = np.zeros_like(weight_tensor)
        if self.r is  None:
            self.r = np.zeros_like(weight_tensor)
        if isinstance(weight_tensor, np.ndarray):
            if self.v.shape != weight_tensor.shape:
                self.v = np.zeros_like(weight_tensor)
            if self.r.shape != weight_tensor.shape:
                self.r = np.zeros_like(weight_tensor)
        self.t += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor #(1,3)
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor * gradient_tensor) 
        v_correct = self.v / (1 - self.mu**self.t)
        r_correct = self.r / (1 - self.rho**self.t)
        tmp =self.learning_rate * (v_correct  / (np.sqrt(r_correct) + np.finfo(np.float64).eps))
        weight_tensor = weight_tensor - tmp
        return weight_tensor