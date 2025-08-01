import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_list = os.listdir(file_path)
        self.file_path = file_path
        with open(label_path,'r',encoding="utf-8") as f:
            self.labels = json.load(f)
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self. shuffle = shuffle
        self.num_epoch = -1
        self.batch_index = 0
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.shuffle_imgs()
        self.show_col = 4

        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        images = []
        labels = []
        s_index = self.batch_index
        n = len(self.file_list)
        for i in range(self.batch_size):
            s_index +=1
            if s_index ==1:
                self.num_epoch+=1
            if s_index >= n:
                s_index = 0
                self.shuffle_imgs()
            image_file = self.file_list[s_index]
            labels.append(self.labels[image_file.split('.')[0]])
            image_data = np.load(os.path.join(self.file_path,image_file))
            image_data=resize(image_data, self.image_size, order=0)
            image_data=self.augment(image_data)
            images.append(image_data)
        self.batch_index = s_index
        return np.stack(images,axis=0), np.stack(labels,axis=0)
    
    def shuffle_imgs(self):
        if self.shuffle:
            random.shuffle(self.file_list)
    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1)  
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0) 
        if self.rotation:
            k_r = np.random.choice([1,2,3]) 
            img = np.rot90(img, k=k_r,axes=(0, 1))
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.num_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        result = "out of index!"
        if x < len(self.class_dict):
            result = self.class_dict[x]
        return result
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images,labels = self.next()
        number_img = images.shape[0]
        show_row =  number_img // self.show_col + 1 
        _, axs = plt.subplots(show_row, self.show_col, figsize=(16, show_row * 4))
        axs = axs.flatten()
        for i in range(number_img):
            axs[i].imshow(images[i])
            axs[i].set_title(self.class_name(labels[i]))
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()
        


