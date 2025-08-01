import numpy as np
import matplotlib.pyplot as plt
class Checker():
    def __init__(self,resolution,tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.list_length = resolution //tile_size
        
    def draw(self):
        if self.resolution % (self.tile_size*2) ==0: 
            output= np.tile(
                [[0,1],[1,0]],
                (self.list_length // 2+1,self.list_length // 2+1)
                )[:self.list_length,:self.list_length]
            self.output=np.repeat(np.repeat(output, self.tile_size, axis=1), self.tile_size, axis=0)
        else:
            self.output=[]
        return self.output.copy()

    def show(self):
        plt.imshow(self.output,cmap='gray')
        plt.show()

class Circle():
    def __init__(self,resolution,radius,position):
        self.resolution = resolution
        self.radius = radius
        self.x = position[0]
        self.y = position[1]

    def draw(self):
        mx = np.arange(self.resolution)
        my = np.arange(self.resolution)
        mX , mY = np.meshgrid(mx,my) #position of each point
        ma= (mX - self.x)**2 + (mY - self.y)**2 #(x-cx)^2 + (y-cy)^2 = d^2
        matrix =ma <= self.radius**2 #bool matrix.
        self.output =  matrix
        return matrix.astype(bool).copy()

    def show(self):
        plt.imshow(self.output,cmap='gray')
        plt.show()

class Spectrum():
    '''
        (0,0,1) (0,0)
        (1,0,0) (0,254)
        (0,1,1) (254,0)
        (1,1,0) (254,254)
        B channel: x:G down 
    '''
    def __init__(self,resolution):
        self.resolution = resolution
    def draw(self):
        arr_r = np.linspace(0,1,self.resolution)
        rlayer_g = np.tile(arr_r,(self.resolution,1))
        glayer_g = np.tile(arr_r.reshape(-1,1),(1,self.resolution))
        blayer_g = np.flip(rlayer_g,axis=1)
        self.output =  np.stack((rlayer_g,glayer_g,blayer_g),axis=2)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()