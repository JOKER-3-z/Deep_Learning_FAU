class BaseLayer:
    def __init__(self,trainable=False,weight=None):
        self.trainable = trainable
        self.weights = weight
