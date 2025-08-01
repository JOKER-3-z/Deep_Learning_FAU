import numpy as np
class L1_Regularizer():
    def __init__(self,alpha):
        self.alpha = alpha
    def calculate_gradient(self,weights):
        gradient=np.sign(weights)
        return self.alpha *gradient 
    def norm(self,weights):
        return self.alpha * np.sum(np.abs(weights))

class L2_Regularizer():
    def __init__(self,alpha):
        self.alpha = alpha
    def calculate_gradient(self,weights):
        return self.alpha*weights
    def norm(self,weights):
        return self.alpha * np.sum(np.power(weights,2))
