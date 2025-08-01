import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_position = None

    def forward(self, input_tensor):
        """
            input_tensor: (bs, c, h, w)
            output: (bs,c,h-m//sy+1,w-n//sx+1)
        """
        self.input_tensor = input_tensor
        b, c, y, x = input_tensor.shape
        m, n = self.pooling_shape
        ty, tx = self.stride_shape
        new_h = (y - m) // ty + 1
        new_w = (x - n) // tx + 1
        self.output = np.zeros((b, c, new_h, new_w))
        self.max_position = np.zeros((b, c, new_h, new_w, 2), dtype=int)

        for i in range(b):  
            for j in range(new_h):  
                for k in range(new_w): 
                    window = input_tensor[i, :, j * ty:j * ty + m, k * tx:k * tx + n]
                    max_value = np.max(window, axis=(-2, -1))  # get max values
                    #get position of max values
                    flat_indices = np.argmax(window.reshape(c, -1), axis=-1) #flatten first
                    max_indices = np.stack(np.unravel_index(flat_indices, (m, n)), axis=-1)#transfer to 2d position
                    # recoder value and position of each max
                    self.output[i, :, j, k] = max_value
                    self.max_position[i, :, j, k, :] = max_indices
        return self.output

    def backward(self, error_tensor):
        b, c, y, x = self.input_tensor.shape
        ty, tx = self.stride_shape
        grad = np.zeros_like(self.input_tensor)
        new_h, new_w = error_tensor.shape[2], error_tensor.shape[3]

        for i in range(b): 
            for j in range(new_h):  
                for k in range(new_w): 
                    for channel in range(c):
                        j_start, k_start = j * ty, k * tx
                        max_j, max_k = self.max_position[i, channel, j, k]
                        grad[i, channel, j_start + max_j, k_start + max_k] += error_tensor[i, channel, j, k]
        return grad
