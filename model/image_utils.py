#!/usr/bin/env python
import cv2
from google.cloud import storage
import io
from PIL import Image
from scipy.io import loadmat # Library to get mat files from SVHN Dataset
import matplotlib.image as img # Get images and make them matrices from Celeb A
from matplotlib import pyplot as plt # Plot images
from os import listdir # Get jpg images from Celeb A
from os.path import isfile, join # Just to get structure of files
import numpy as np
from tempfile import TemporaryFile # To save matrix of images
import random


def show_1(image): # func to plot image
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

def halve_pic(pics,place):
    
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

def load_images_from_bucket(bucket='inpainting-final-project', path='images/Cars/cars_train/'):
    """
    Loading the images from the Google Cloud bucket
    """
    # Open bucket
    client = storage.Client()
    bucket = client.get_bucket('inpainting-final-project')
    blobs = bucket.list_blobs(prefix='images/Cars/cars_train/')
    images = [] 
    # Append images
    try:
        for blob in blobs:
            blob = bucket.get_blob(blob.name)
            s = blob.download_as_string()
            img = Image.open(io.BytesIO(s))
            
            #resize the image to (64,64,3) and normalize it to between -1 and 1
            resized_img = cv2.resize(np.asarray(img),(64,64))/127.5-1.0
            
            if resized_img.shape == (64,64,3):
                images.append(resized_img)
    except:
        pass
    return np.asarray(images)


def add_noise(x, portion, amplitude):
    """
    Add random integer noise to self.x.
    :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                    then 1000 samples will be noise-injected.
    :param amplitude: An integer scaling factor of the noise.
    :return added: dataset with noise added
    """
    # TODO: Implement the add_noise function. Remember to record the
    # boolean value is_add_noise. You can try uniform noise or Gaussian
    # noise or others ones that you think appropriate.
    # raise NotImplementedError
    
    channels = 3
    num_of_samples = len(x)
    
    for i in range(num_of_samples):
        #in each sample, we need to shift for each channel
        random_boolean = np.random.choice(a=[True, False], size=1, p=[portion, 1-portion])
    
        if random_boolean == True:
            
            for j in range(channels):
    
                mean = 0
                std = 0.01
                noise = amplitude * np.random.normal(mean, std, x[i,:,:,j].shape)
                print(noise)
                x[i,:,:,j] += noise
    return x 


class ImageCollector():

    def __init__(self, x):
        """
        """
        self.x = x
        self.num_of_samples, self.height, self.width, self.channels = x.shape
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False

        self.x_aug = x

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        x = self.x_aug

        batch_count = 1
        total_batches = self.num_of_samples // batch_size
        condition = True
        while condition:
            if (batch_count < total_batches):
                batch_count += 1
                x_batch = x[(batch_count-1)*batch_size:batch_count*batch_size,:,:,:]
                yield x_batch

            else:
                if not shuffle:
                    condition = False
                else:
                    permutation = np.random.permutation(self.num_of_samples)
                    x = x[permutation,:,:,:]
                    batch_count = 1


    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        #images = images.reshape(128, 3, 32, 32).transpose(0,2,3,1)
        i = 0
        fig, axes1 = plt.subplots(4,4,figsize=(10,10))
        for j in range(4):
            for k in range(4):
                i += 1
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(images[i:i+1][0])

