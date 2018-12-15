#!/usr/bin/env python

from scipy.io import loadmat # Library to get mat files from SVHN Dataset
import matplotlib.image as img # Get images and make them matrices from Celeb A
from matplotlib import pyplot as plt # Plot images
from os import listdir # Get jpg images from Celeb A
from os.path import isfile, join # Just to get structure of files
import numpy as np
from tempfile import TemporaryFile # To save matrix of images
import random


def show(image): # func to plot image
    plt.imshow(image)
    plt.show()
    return None

def cut(images): # cropping images
    images_cr = []
    for image in images:
        images_cr.append(image[45:173,25:153,:])
                            # [77:141,57:121,:] for 64 X 64
    return images_cr

def delete_pixels(pics, portion):

    images = np.copy(pics)
    mask = np.ones(images.shape)
    out = []
    im = 0
    for image in images: # For every image
        image.setflags(write=1)
        for i in range(0,len(images[0])):
            for j in range(0,len(images[0][0])): # For every pixel
                p = random.uniform(0,1) # Random p
                if p > 1 - portion: # Probability of getting noise is the portion.
                    image[i,j] = [0]*3 # Delete the pixel
                    mask[im][i,j] = 0
        out.append(image) # Get array of new images
        im += 1
        
    return np.array(out),mask

def center_square(pics):
    
    out =[]
    images = np.copy(pics)
    mask = np.ones(images[0].shape)
    dim1 = len(mask)
    dim2 = len(mask[0])
    
    mask[int(dim1/4):int(3*dim1/4),int(dim2/4):int(3*dim2/4),:] = 0
    
    for image in images: # For every image
        image.setflags(write=1)
        for i in range(0,len(images[0])):
            for j in range(0,len(images[0][0])): # For every pixel
                if mask[i,j].any() == 0: # Use the mask to 
                    image[i,j] = [0]*3 # delete the pixel
        out.append(image) # Get array of new images
    
    return np.array(out),mask

def center_square(pics,place):
    
    out =[]
    images = np.copy(pics)
    mask = np.ones(images[0].shape)
    dim1 = len(mask)
    dim2 = len(mask[0])
    
    if place.lower() == 'up':
        mask[:int(dim2/2),:,:] = 0
    elif place.lower() == 'down':
        mask[int(dim2/2):,:,:] = 0
    elif place.lower() == 'right':
        mask[:,int(dim2/2):,:] = 0
    elif place.lower() == 'left':
        mask[:,:int(dim2/2),:] = 0
    else:
        raise ValueError('That makes no sense.')
        
    
    for image in images: # For every image
        image.setflags(write=1)
        for i in range(0,len(images[0])):
            for j in range(0,len(images[0][0])): # For every pixel
                if mask[i,j].any() == 0: # Use the mask to 
                    image[i,j] = [0]*3 # delete the pixel
        out.append(image) # Get array of new images
    
    return np.array(out),mask

