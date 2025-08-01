import math
import numpy as np
from Layers.Base import BaseLayer
class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernel):
        self.trainable = True
        self.kernel_shape = (num_kernel, *convolution_shape)
        self.weights = np.random.uniform(0, 1, self.kernel_shape)
        self.bias = np.random.uniform(0, 1, (num_kernel, 1))
        self.stride = stride_shape
        self.num_kernel = num_kernel
        self.convolution_shape = convolution_shape
        self._gw = None
        self._gb = None
        self.optimizer = None

    @property
    def gradient_weights(self):
        return self._gw

    @property
    def gradient_bias(self):
        return self._gb

    def initialize(self,weights_initializer_methods, bias_initializer_methods):
        fan_in = math.prod(self.convolution_shape)
        fan_out = fan_in / self.convolution_shape[0] * self.num_kernel
        self.weights = weights_initializer_methods.initialize(self.weights.shape,fan_in,fan_out)
        self.bias = bias_initializer_methods.initialize(self.bias.shape,fan_in,fan_out)

    def conv_1D(self, input, output_shape,stride):
        """
            input: (c, y)
            output_shape: (num_kernel, (y-m)/s)
            output: (num_kernel, (y-m)/s)
        """
        num_kernel, output_width = output_shape
        output = np.zeros(output_shape)

        for k in range(num_kernel):
            kernel = self.weights[k]
            bias = self.bias[k]
            for i in range(0, input.shape[-1] - kernel.shape[-1] + 1, stride):
                region = input[:, i:i + kernel.shape[-1]]
                output[k, i // stride] = np.sum(region * kernel) + bias

        return output

    def conv_2D(self, input, output_shape, strides):
        """
            input: (c, y, x)
            output_shape: (num_kernel, (y-m)/s_y, (x-n)/s_x)
            strides: (stride_y, stride_x)
            kernel: (num_kernel, c, y, x)
            output: (num_kernel, (y-m)/s_y, (x-n)/s_x)
        """
        stride_y, stride_x = strides
        num_kernel, output_height, output_width = output_shape
        output = np.zeros(output_shape)

        for k in range(num_kernel):
            kernel = self.weights[k]
            bias = self.bias[k]
            for i in range(0, input.shape[1] - kernel.shape[1]+1, stride_y):
                for j in range(0, input.shape[2] - kernel.shape[2]+1, stride_x):
                    region = input[:, i:i + kernel.shape[1], j:j + kernel.shape[2]]
                    output[k, i // stride_y, j // stride_x] = np.sum(region * kernel) + bias
        return output

    def forward(self, input_tensor):
        """
        input_tensor: (b, c, y) / (b, c, y, x)
        output_tensor: (b, num_kernel, (y-m)//s+1) /(b, num_kernel, (y-m)//s_y+1, (x-n)//s_x+1)
        """
        n = len(input_tensor.shape)
        output = []
        self.input_tensor = input_tensor
        if n == 3:
            # 1D convolution
            b, c, y = input_tensor.shape
            py = (self.kernel_shape[2]-1)//2
            py_end = py
            if self.kernel_shape[2]%2 ==0 :
                    py_end +=1
            self.padding = [(0,0),  (0,0), (py,py_end)]
            self.pad_input_tensor = np.pad(input_tensor, self.padding, mode='constant', constant_values=0)
            output_shape = (self.num_kernel, (self.pad_input_tensor.shape[-1] - self.kernel_shape[2]) // self.stride[0]+1)
            for i in range(b):
                single_input = self.pad_input_tensor[i]
                output.append(self.conv_1D(single_input, output_shape,self.stride[0]))
        elif n == 4:
            # 2D convolution
            b, c, y, x = input_tensor.shape
            if isinstance(self.stride, tuple):
                self.stride_y, self.stride_x = self.stride
            else:
                self.stride_y, self.stride_x = self.stride, self.stride
            px,py = (self.kernel_shape[3]-1)//2,(self.kernel_shape[2]-1)//2
            px_end,py_end = px,py
            if self.kernel_shape[3]%2 ==0 :
                    px_end +=1
            if  self.kernel_shape[2]%2 ==0 :
                    py_end +=1
            self.padding = [(0, 0), (0, 0), (py, py_end), (px, px_end)]
            self.pad_input_tensor = np.pad(input_tensor, self.padding, mode='constant', constant_values=0)

            output_shape = (self.num_kernel,
                            (self.pad_input_tensor.shape[-2]-self.weights.shape[-2])//self.stride_y+1,
                            (self.pad_input_tensor.shape[-1]-self.weights.shape[-1])//self.stride_x+1)
            for i in range(b):
                single_input = self.pad_input_tensor[i]
                output.append(self.conv_2D(single_input, output_shape, [self.stride_y, self.stride_x]))

        self.output = np.array(output)
        return self.output
    
    def backward_1D(self, error_tensor):
        db = np.sum(error_tensor, axis=(0, 2)).reshape(-1,1)
        dw = np.zeros(self.weights.shape)
        #padded_err = np.pad(error_tensor, [(0, 0), (0, 0), (self.kernel_shape[-1] - 1, self.kernel_shape[-1] - 1)], mode='constant', constant_values=0)
        for k in range(self.num_kernel):
            for i in range(0,(self.pad_input_tensor.shape[-1]-error_tensor.shape[-1] )//self.stride[0]-1):
                dw[k,:,i] = np.sum(self.pad_input_tensor[:,:,i:i+error_tensor.shape[2]] * error_tensor[:,k:k+1,:],axis=(0,2))
        self._gb = db
        self._gw = dw
        d_input = np.zeros(self.input_tensor.shape)#(b,c,w)
        flipped_weights=np.flip(self.weights, axis=-1)
        for n in range(self.input_tensor.shape[0]):  # 遍历批次
            for c_in in range(self.input_tensor.shape[1]):  # 遍历输入通道
                for c_out in range(self.weights.shape[0]):  # 遍历输出通道
                    # 卷积核与步幅处理
                    flipped_kernel = flipped_weights[c_out, c_in, :]
                    for i in range(error_tensor.shape[-1]):  # 按照步幅更新 d_input
                        start = i * self.stride[0]
                        end = start + self.weights.shape[-1]
                        if end <= self.input_tensor.shape[-1]:  # 确保范围合法
                            d_input[n, c_in, start:end] += error_tensor[n, c_out, i] * flipped_kernel
        return d_input

    def backward_2D(self, error_tensor):

        db = np.sum(error_tensor, axis=(0,2,3)).reshape(-1,1)
        dw = np.zeros_like(self.weights)
        grad_input = np.zeros_like(self.pad_input_tensor)
        rotated_kernels = np.flip(self.weights, axis=(2, 3))
        # 卷积核梯度计算
        for n in range(error_tensor.shape[0]):  
            for k in range(self.num_kernel):  
                for c in range(self.kernel_shape[1]):  
                    for i in range(error_tensor.shape[2]):  
                        for j in range(error_tensor.shape[3]):  
                            start_i,end_i = i * self.stride_y,i * self.stride_y + self.kernel_shape[2]
                            start_j,end_j = j * self.stride_x,j * self.stride_x + self.kernel_shape[3]

                            if end_i <= self.pad_input_tensor.shape[2] and end_j <= self.pad_input_tensor.shape[3]:
                                input_slice = self.pad_input_tensor[n, c, start_i:end_i, start_j:end_j]
                                error_value = error_tensor[n, k, i, j]
                                dw[k, c, :, :] += input_slice * error_value
        """
        print(error_tensor.shape)
        print(self.kernel_shape)
        print(self.weights.shape)
        print(self.pad_input_tensor.shape)
        print(self.input_tensor.shape)
        print(self.padding[3])
        """
        # 输入梯度计算
        padded_error = np.pad(
            error_tensor,
            [(0, 0), (0, 0), (self.kernel_shape[2] - 1, self.kernel_shape[2] - 1), (self.kernel_shape[3] - 1, self.kernel_shape[3] - 1)],
            mode="constant",
            constant_values=0,
        )
        # 反向传播计算梯度
        for n in range(error_tensor.shape[0]):
            for k in range(self.num_kernel):
                for c in range(self.kernel_shape[1]):
                    for i in range(error_tensor.shape[2]):
                        for j in range(error_tensor.shape[3]):
                            h_start,h_end = i * self.stride_y, +i * self.stride_y + self.kernel_shape[2]
                            w_start,w_end = j * self.stride_x,j * self.stride_x + self.kernel_shape[3]
                            grad_input[n, c, h_start:h_end, w_start:w_end] += self.weights[k, c, :, :] * error_tensor[n, k, i, j]

        # 去除填充部分，恢复输入形状
        py, py_end = self.padding[2]
        px, px_end = self.padding[3]
        grad_input = grad_input[:, :, py:-py_end if py_end != 0 else None, px:-px_end if px_end != 0 else None]

        # 保存梯度
        self._gb = db
        self._gw = dw
        return grad_input

    def backward(self, error_tensor):
        """
            error_tensor: (b, num_kernel, output_height, output_width) / (b, num_kernel, output_length)
        """
        if len(self.input_tensor.shape) == 3:  # 1D convolution backward
            gradin_input =  self.backward_1D(error_tensor)
        elif len(self.input_tensor.shape) == 4:  # 2D convolution backward
            gradin_input =  self.backward_2D(error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self._gw)
            self.bias = self.optimizer.calculate_update(self.bias,self._gb)
        return gradin_input