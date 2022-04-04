import copy
import numpy as np
import random

import cv2
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Input: s - strength of color distortion
# Optional: transforms - list of transforms to be applied in addition to color distortion
# Returns a transforms for creating transforms.Compose() to initialize a torchvision datasets
# https://arxiv.org/pdf/2002.05709.pdf
def get_color_distortion(s=1.0, p=0.8):
    
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=p)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    transform = [rnd_color_jitter, rnd_gray]

    return transform

# k: kernel size. Should be odd
# https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size
# The size of the kernel should normally be selected large enough so that the kernel coefficients of the border rows and columns contribute very little to the sum of coefficients (0 at the edge). 
# By selecting a kernel size parameters six times the standard deviation the border parameters will be 1% or lower than the center parameter.
def get_gaussian_blur(k=21, p=0.8):
    
    blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=k)], p=p)
    return [blur]

# p: probability of noise
# Note: this function must be applied on a tensor and not PIL image (i.e. when doing transforms.Compose, you should add this after transforms.ToTensor())
def get_gaussian_noise(mean = 0., std = 1., p=0.8):

    noise = transforms.RandomApply([GaussianNoise(mean, std)], p=p )
    return [noise]


##################################### Manual Implementation of Functions ################################################

def gaussian_blur_manual(image):
    number = random.choice([0,1])
    src = np.float32(image)

    # apply gaussian blur
    if number == 1:
        height_gauss = 0.1 *  image.shape[0]
        width_gauss = 0.1 * image.shape[1]

        # Gaussian blur function
        if (image.shape[0] != image.shape[1]):
            minimum = min(height_gauss, width_gauss)
            height_gauss = minimum
            width_gauss = minimum

        # To pass into function the values have to be Odd
        height_gauss = int(height_gauss)
        width_gauss = int(width_gauss)

        if height_gauss % 2 == 0:
            height_gauss += 1
        if width_gauss % 2 == 0:
            width_gauss += 1
        ksize = (height_gauss, width_gauss)
        sigma_x = random.uniform(0.1,2.0)
        sigma_y = random.uniform(0.1,2.0)
        dst = cv2.GaussianBlur(src,ksize=ksize,sigmaX=sigma_x,sigmaY=sigma_y,borderType=cv2.BORDER_DEFAULT)
    
    # don't apply anything
    elif number == 0:
        dst = src

    return dst

# p: probability of noise
def gaussian_noise_manual(img, p=0.5):
    row , col, channel = img.shape
    gaussian_noise = np.zeros((row , col, channel), np.uint8)
    threshold = 1 - p
    for i in range(row):
        for j in range(col):
            rdn = random.random()
            if rdn < p:
                gaussian_noise[i][j] = 0
            elif rdn > threshold:
                gaussian_noise[i][j] = 255
            else:
                gaussian_noise[i][j] = img[i][j]
    return gaussian_noise


########################################################################################################


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)