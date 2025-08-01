import numpy as np
class Constant:
    def __init__(self,initia_member=0.1):
        self.initia_member=initia_member
    def initialize(self,weights_shape,fan_in, fan_out):
        return np.full(weights_shape,self.initia_member)
        

class UniformRandom():
    def initialize(self,weights_shape,fan_in, fan_out):
        return np.random.uniform(0,1+np.finfo(float).eps,weights_shape)#控制增量

class Xavier ():
    def initialize(self,weights_shape,fan_in, fan_out):
        std_value = np.sqrt(2/(fan_in+fan_out))
        return np.random.normal(loc=0,scale=std_value,size=weights_shape)

class He ():
    def initialize(self,weights_shape,fan_in, fan_out):
        std_value = np.sqrt(2/fan_in)
        return np.random.normal(loc=0,scale=std_value,size=weights_shape)