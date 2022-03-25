import copy
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Input: s - strength of color distortion
# Optional: transforms - list of transforms to be applied in addition to color distortion
# Returns a transforms.Compose for initializing a torchvision datasets
def get_color_distortion(transform: list, s=1.0):
    if transform is None:
        transform = [transforms.ToTensor()]
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray, *transform])

    return color_distort
    
def get_gaussian_blur(transform: list):
    pass

def get_gaussian_noise(transform: list):
    pass
   