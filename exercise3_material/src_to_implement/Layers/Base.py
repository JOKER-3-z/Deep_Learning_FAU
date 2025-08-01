class BaseLayer:
    def __init__(self,trainable=False,weight=None,testing_phase = False):
        self.trainable = trainable
        if weight is not None:
            self.weights = weight
        self.testing_phase=testing_phase
