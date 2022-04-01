import copy
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Input: s - strength of color distortion
# Optional: transforms - list of transforms to be applied in addition to color distortion
# Returns a transforms array for creating transforms.Compose() to initialize a torchvision datasets
def get_color_distortion(transform: list, s=1.0):
    if transform is None:
        transform = [transforms.ToTensor()]
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = [rnd_color_jitter, rnd_gray, *transform]

    return color_distort
    
def get_gaussian_blur(transform: list):
    pass

# p: probability of noise
def get_gaussian_noise(transform: list, img, p=0.05):
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
                gaussian_noise[i][j] = image[i][j]
    return gaussian_noise
