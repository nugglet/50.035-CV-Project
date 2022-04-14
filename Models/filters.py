import copy
import numpy as np
import random

import cv2
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Input: s - strength of color distortion
# Returns a list of transforms for creating transforms.Compose() to initialize a torchvision datasets
# https://arxiv.org/pdf/2002.05709.pdf
def get_color_distortion(s=1.0, p=0.8):
    
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=p)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    transform = [rnd_color_jitter, rnd_gray]

    return transform

# k: kernel size. Should be odd
# https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size
# The size of the kernel should normally be selected large enough so that the kernel coefficients of the 
# border rows and columns contribute very little to the sum of coefficients (0 at the edge). 
# By selecting a kernel size parameters six times the standard deviation the border parameters will be 
# 1% or lower than the center parameter.
def get_gaussian_blur(k=21, p=0.8):
    
    blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=k)], p=p)
    return [blur]

# p: probability of noise
# Note: this function must be applied on a tensor and not PIL image 
# (i.e. when doing transforms.Compose, you should add this after transforms.ToTensor())
def get_gaussian_noise(mean = 0., std = 1., p=0.8):

    noise = transforms.RandomApply([GaussianNoise(mean, std)], p=p )
    return [noise]


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)