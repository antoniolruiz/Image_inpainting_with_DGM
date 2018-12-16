import tensorflow as tf
import numpy as np

from scipy.signal import convolve2d
import scipy.misc
import argparse
import os
from glob import glob
import pyamg

#possion blending 

# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    
    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target


def test():
    img_mask = np.asarray(PIL.Image.open('./testimages/test1_mask.png'))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open('./testimages/test1_src.png'))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open('./testimages/test1_target.png'))
    img_target.flags.writeable = True
    img_ret = blend(img_target, img_source, img_mask, offset=(40,-30))
    img_ret = PIL.Image.fromarray(np.uint8(img_ret))
    img_ret.save('./testimages/test1_ret.png')



class ModelInpaint():
    def __init__(self, modelfilename, iters, l, learning_rate, momentum,
                 model_name='dcgan',
                 gen_input='z:0', gen_output='Tanh:0', gen_loss='Mean_2:0',
                 disc_input='real_images:0', disc_output='Sigmoid:0',
                 z_dim=100, batch_size=128):
        """
        Model for Semantic image inpainting.
        Loads frozen weights of a GAN and create the graph according to the
        loss function as described in paper
        Arguments:
            modelfilename - tensorflow .pb file with weights to be loaded
            config - training parameters: lambda_p, nIter
            gen_input - node name for generator input
            gen_output - node name for generator output
            disc_input - node name for discriminator input
            disc_output - node name for discriminator output
            z_dim - latent space dimension of GAN
            batch_size - training batch size
        """

      

        self.batch_size = batch_size
        self.z_dim = z_dim
#         self.graph, self.graph_def = ModelInpaint.loadpb(modelfilename,
#                                                          model_name)

        self.graph = ModelInpaint.restore_graph()

        self.gi = self.graph.get_tensor_by_name(gen_input)
        self.go = self.graph.get_tensor_by_name(gen_output)
        self.gl = self.graph.get_tensor_by_name(gen_loss)
        self.di = self.graph.get_tensor_by_name(disc_input)
        self.do = self.graph.get_tensor_by_name(disc_output)
        
        #new edits 
        self.training = self.graph.get_tensor_by_name('inputs/is_training:0')

        self.image_shape = self.go.shape[1:].as_list()

        self.l = l
        self.lr = learning_rate
        self.iters = iters
        self.momentum = momentum

        self.sess = tf.Session(graph=self.graph)

        self.init_z()

    def init_z(self):
        """Initializes latent variable z"""
        self.z = np.random.randn(self.batch_size, self.z_dim)

    def sample(self, z=None):
        """GAN sampler. Useful for checking if the GAN was loaded correctly"""
        if z is None:
            z = self.z
        #new edits 
        self.sess.run(tf.global_variables_initializer())
        sample_out = self.sess.run(self.go, feed_dict={self.gi: z, self.training: False})
        return sample_out

    def preprocess(self, images, imask, useWeightedMask = True, nsize=7):
        """Default preprocessing pipeline
        Prepare the data to be fed to the network. Weighted mask is computed
        and images and masks are duplicated to fill the batch.
        Arguments:
            image - input image
            mask - input mask
        Returns:
            None
        """
        images = ModelInpaint.imtransform(images)
        if useWeightedMask:
            mask = ModelInpaint.createWeightedMask(imask, nsize)
        else:
            mask = imask
        mask = ModelInpaint.create3ChannelMask(mask)
        
        bin_mask = ModelInpaint.binarizeMask(imask, dtype='uint8')
        self.bin_mask = ModelInpaint.create3ChannelMask(bin_mask)

        self.masks_data = np.repeat(mask[np.newaxis, :, :, :],
                                    self.batch_size,
                                    axis=0)

        #Generate multiple candidates for completion if single image is given
        if len(images.shape) is 3:
            self.images_data = np.repeat(images[np.newaxis, :, :, :],
                                         self.batch_size,
                                         axis=0)
        elif len(images.shape) is 4:
            #Ensure batch is filled
            num_images = images.shape[0]
            self.images_data = np.repeat(images[np.newaxis, 0, :, :, :],
                                         self.batch_size,
                                         axis=0)
            ncpy = min(num_images, self.batch_size)
            self.images_data[:ncpy, :, :, :] = images[:ncpy, :, :, :].copy()

    def postprocess(self, g_out, blend = True):
        """Default post processing pipeline
        Applies poisson blending using binary mask. (default)
        Arguments:
            g_out - generator output
            blend - Use poisson blending (True) or alpha blending (False)
        """
        images_out = ModelInpaint.iminvtransform(g_out)
        images_in = ModelInpaint.iminvtransform(self.images_data)

        if blend:
            for i in range(len(g_out)):
                images_out[i] = ModelInpaint.poissonblending(
                    images_in[i], images_out[i], self.bin_mask
                )
        else:
            images_out = np.multiply(images_out, 1-self.masks_data) \
                         + np.multiply(images_in, self.masks_data)

        return images_out

    def build_inpaint_graph(self):
        """Builds the context and prior loss objective"""
        with self.graph.as_default():
            self.masks = tf.placeholder(tf.float32,
                                        [None] + self.image_shape,
                                        name='mask')
            self.images = tf.placeholder(tf.float32,
                                         [None] + self.image_shape,
                                         name='images')
            self.context_loss = tf.reduce_sum(
                    tf.contrib.layers.flatten(
                        tf.abs(tf.multiply(self.masks, self.go) -
                               tf.multiply(self.masks, self.images))), 1
                )

            self.perceptual_loss = self.gl
            #self.perceptual_loss = tf.cast(self.gl, tf.float32)
#             print(self.perceptual_loss)
#             print(self.l)
            
            self.inpaint_loss = self.context_loss + self.l*self.perceptual_loss
            self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)

    def inpaint(self, image, mask, blend=True):
        """Perform inpainting with the given image and mask with the standard
        pipeline as described in paper. To skip steps or try other pre/post
        processing, the methods can be called seperately.
        Arguments:
            image - input 3 channel image
            mask - input binary mask, single channel. Nonzeros values are 
                   treated as 1
            blend - Flag to apply Poisson blending on output, Default = True
        Returns:
            post processed image (merged/blneded), raw generator output
        """
        self.build_inpaint_graph()
        self.preprocess(image, mask)

        imout = self.backprop_to_input()

        return self.postprocess(imout, blend), imout

    def backprop_to_input(self, verbose=True):
        """Main worker function. To be called after all initilization is done.
        Performs backpropagation to input using (accelerated) gradient descent
        to obtain latent space representation of target image
        Returns:
            generator output image
        """
        v = 0
        for i in range(self.iters):
            out_vars = [self.inpaint_loss, self.inpaint_grad, self.go]
            in_dict = {self.masks: self.masks_data,
                       self.gi: self.z,
                       self.images: self.images_data,
                       self.training: False}

            loss, grad, imout = self.sess.run(out_vars, feed_dict=in_dict)

            v_prev = np.copy(v)
            v = self.momentum*v - self.lr*grad[0]
            self.z += (-self.momentum * v_prev +
                       (1 + self.momentum) * v)
            self.z = np.clip(self.z, -1, 1)

            if verbose:
                print('Iteration {}: {}'.format(i, np.mean(loss)))

        return imout

    @staticmethod
    def loadpb(filename, model_name='dcgan'):
        """Loads pretrained graph from ProtoBuf file
        Arguments:
            filename - path to ProtoBuf graph definition
            model_name - prefix to assign to loaded graph node names
        Returns:
            graph, graph_def - as per Tensorflow definitions
        """
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                op_dict=None,
                                producer_op_list=None,
                                name=model_name)

        return graph, graph_def
    
    
    def restore_graph():
        with tf.Session() as sess:
            loader = tf.train.import_meta_graph('../CelebA/checkpoints/DCGAN.ckpt.meta')
            loader.restore(sess, '../CelebA/checkpoints/DCGAN.ckpt')
            graph = tf.get_default_graph()
        return graph

    @staticmethod
    def imtransform(img):
        """Helper: Rescale pixel value ranges to -1 and 1"""
        return np.array(img) / 127.5-1

    @staticmethod
    def iminvtransform(img):
        """Helper: Rescale pixel value ranges to 0 and 1"""
        return (np.array(img) + 1.0) / 2.0

    @staticmethod
    def poissonblending(img1, img2, mask):
        """Helper: interface to external poisson blending"""
        return blend(img1, img2, 1 - mask)

    @staticmethod
    def createWeightedMask(mask, nsize=7):
        """Takes binary weighted mask to create weighted mask as described in 
        paper.
        Arguments:
            mask - binary mask input. numpy float32 array
            nsize - pixel neighbourhood size. default = 7
        """
        ker = np.ones((nsize,nsize), dtype=np.float32)
        ker = ker/np.sum(ker)
        wmask = mask * convolve2d(mask, ker, mode='same', boundary='symm')
        return wmask

    @staticmethod
    def binarizeMask(mask, dtype=np.float32):
        """Helper function, ensures mask is 0/1 or 0/255 and single channel
        If dtype specified as float32 (default), output mask will be 0, 1
        if required dtype is uint8, output mask will be 0, 255
        """
        assert(np.dtype(dtype) == np.float32 or np.dtype(dtype) == np.uint8)
        bmask = np.array(mask, dtype=np.float32)
        bmask[bmask>0] = 1.0
        bmask[bmask<=0] = 0
        if dtype == np.uint8:
            bmask = np.array(bmask*255, dtype=np.uint8)
        return bmask
    
    @staticmethod
    def create3ChannelMask(mask):
        """Helper function, repeats single channel mask to 3 channels"""
        assert(len(mask.shape)==2)
        return np.repeat(mask[:,:,np.newaxis], 3, axis=2)

def gen_mask(maskType):
    
    
#     image_shape = [args.imgSize, args.imgSize]
    image_shape = [64, 64]
    if maskType == 'random':
        fraction_masked = 0.2
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
    elif maskType == 'center':
        scale = 0.25
        assert(scale <= 0.5)
        mask = np.ones(image_shape)
        sz = 64
        l = int(64*scale)
        u = int(64*(1.0-scale))
        mask[l:u, l:u] = 0.0
    elif maskType == 'left':
        mask = np.ones(image_shape)
        c = args.imgSize // 2
        mask[:, :c] = 0.0
#     elif maskType == 'file':
#         mask = loadmask(args.maskfile, args.maskthresh)
    else:
        assert(False)
    return mask


def loadmask(filename, thresh=128):
    immask = scipy.misc.imread(filename, mode='L')
    image_shape = [args.imgSize, args.imgSize]
    mask = np.ones(image_shape)
    mask[immask < 128] = 0
    mask[immaks >= 128] = 1
    return mask

def loadimage(filename):
    img = scipy.misc.imread(filename, mode='RGB').astype(np.float)
    return img


def saveimages(outimages, outdir, prefix='samples'):
    numimages = len(outimages)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in range(numimages):
        filename = '{}_{}.png'.format(prefix, i)
        filename = os.path.join(outdir, filename)
        scipy.misc.imsave(filename, outimages[i, :, :, :])
