import math, cv2
import numpy as np
from functools import reduce
import operator
from .base import AbstractPCNN

class mPCNN(object):
    
    NAME = 'mPCNN'
    
    def __init__(self, images, parameters):
        self.images = [cv2.normalize(i.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) for i in images]
        self.input_dim = self.images[0].shape
        self.Y = np.zeros(self.input_dim)
        self.U = self.Y
        self.T = np.ones(self.input_dim)
        self.a_T = parameters['a_T']
        self.v_T = parameters['v_T']
        self.lvl_factor = parameters['lvl_factor']
        self.kernel = np.array([
            [0.1091, 0.1409, 0.1091],
            [0.1409, 0, 0.1409],
            [0.1091, 0.1409, 0.1091]
        ])
        self.B = parameters['beta']
        if len(self.B) != len(self.images):
            raise ValueError('Number of beta should be equal to the number of images')
        
    
    def __repr__(self):
        return 'mPCNN Model - Input shape : {}'.format(self.input_dim)
        
    def update_threshold(self) -> None:
        """
        Update the threshold value
        """
        self.T = math.exp(-self.a_T) * self.T + self.v_T * self.Y
        return
    
    def compute_feeding_functions(self):
        W = cv2.filter2D(self.Y.astype('float'), -1, 
                            self.kernel, 
                            borderType=cv2.BORDER_REFLECT_101)
        feeding_functions = [W for i in range(len(self.images))]
        return feeding_functions
    
    def compute_internal_activation(self):
        feeding_functions = self.compute_feeding_functions()
        H = [i + j for (i, j) in zip(feeding_functions, self.images)]
        ops = [(1 + b * h) for (b, h) in zip(self.B, H)]
        self.U = reduce(operator.mul, ops) + self.lvl_factor
        return

    def do_iteration(self):
        Cn = 0
        while True:
            self.compute_internal_activation()
            self.Y = np.where(self.U > self.T, 1, 0)
            self.update_threshold()
            Cn += np.sum(self.Y)
            Sum = np.sum(self.Y) + Cn
            if Sum < self.input_dim[0]**2:
                self.compute_internal_activation()
                self.Y = np.where(self.U > self.T, 1, 0)
                self.update_threshold()
            else:
                self.U = (self.U - np.min(self.U)) / (np.max(self.U) - np.min(self.U))
                break