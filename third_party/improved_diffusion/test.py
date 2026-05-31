import torch
import numpy as np
import cv2


import argparse

# image = cv2.imread('benign (1).png')
# print(image.shape)


# ckpt = OrderedDict:335 ,  model parameters
# 'cifar10_uncond_50M_500K'  50M: param size; 500K: iteration times
ckpt = torch.load("./scripts/cifar10_uncond_50M_500K.pt", map_location='cpu')
print(ckpt)