import numpy as np
 
class CrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.eps= np.finfo(np.float64).eps
    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        # if y==1, sum(-ln(y+eps),batchsize)
        eps= np.finfo(np.float64).eps
        self.loss = -np.sum(np.log(self.prediction_tensor[label_tensor==1]+self.eps),axis=0)
        return self.loss

    def backward(self,label_tensor):
        # 梯度计算公式： grad = -y/(y~ + eps)
        grad = -label_tensor/(self.prediction_tensor + self.eps)
        return grad