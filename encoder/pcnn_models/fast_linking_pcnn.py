from .base import AbstractPCNN
from .utils import gaussian_kernel
from scipy.ndimage import convolve
from scipy.ndimage import measurements
import numpy as np
import cv2, torch

class FLSCM(AbstractPCNN):
    
    NAME = 'FLSCM'
    
    def __init__(self, S, parameters):
        AbstractPCNN.__init__(self, S)
        self.f = parameters['f']
        self.S = (self.S - np.min(self.S)) / (np.max(self.S) - np.min(self.S))
        self.Sm = np.max(self.S)
        self.T = self.T * self.Sm
        self.F = self.S
        self.YY, self.U = self.Y, self.Y
        self.dT = 1/255
        self.v_t = parameters['v_t']
        lap_kernel = np.array([[1,1,1],[1,-8, 1], [1, 1, 1]])
        LAP = abs(convolve(self.S, lap_kernel, mode='mirror'))
        self.beta = 2.3 * LAP
    
    def update_linking(self):
        w = gaussian_kernel(size=5, sigma=2.)
        self.L = convolve(self.Y, w, mode='mirror')
        return

    def update_threshold(self):
        self.T = self.T - self.dT + self.v_t * self.Y
        return

    def compute_internal_activation(self):
        self.U = self.f * self.U + self.F * (1 + self.beta * self.L)
        return

    def do_iteration(self):
        self.update_linking()
        self.compute_internal_activation()
        self.Y = np.where(self.U > self.T, 1, 0)
        self.YY = self.YY + self.Y
        self.update_threshold()
        return self.YY
    

class FL_Signature(FLSCM):
    
    NAME = 'FLSCM_sign'
    
    def __init__(self, S, parameters):
        super().__init__(S, parameters)
        self.signature = []
        # self.weight = round(self.S[self.S.shape[0]//2][self.S.shape[1]//2], 3)
        
    def do_iteration(self):
        total_count = 0
        n = 1
        while total_count != self.S.shape[0]**2:
            self.update_linking()
            self.compute_internal_activation()
            self.Y = np.where(self.U > self.T, 1, 0)
            self.YY = self.YY + self.Y
            total_count += self.Y.sum()
            self.signature.append(self.Y.sum())
            self.update_threshold()
            n += 1