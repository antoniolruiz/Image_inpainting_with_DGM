#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

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

