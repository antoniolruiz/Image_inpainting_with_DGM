# Semantic Image Inpainting with a Novel Twist
Final project for the course of Deep Learning at Columbia University based on Semantic Image Inpainting With Deep Generative Models by Raymond A. Yeh*, Chen Chen*, Teck Yian Lim, Alexander G. Schwing, Mark Hasegawa-Johnson, Minh N. Do. https://arxiv.org/pdf/1607.07539.pdf

Semantic Image Inpainting with a Novel Twist
Ling He, Gerardo Antonio Lopez Ruiz, and Andrea Navarrete Rivera

## Overview

Implementation of DCGAN and inpainting model. 

## Dependencies
 - Tensorflow >= 1.0
 - scipy + PIL/pillow (image io)
 - pyamg (for Poisson blending)
 - Tested to work for Python 3

## Files

- `SVHN train DataSet w labels.ipynb`
- `Run_train_Cars.ipynb`
- `Run_train_CelebA.ipynb`

# Functions of each file in the project:

- `model/dcgan.py`: Includes the class of the DCGAN network with all the layer functions.
- `model/image_utils.py`: Includes functions to preprocess and manipulate the images.
- `model/inpainting.py`: Class for inpainting model which includes restoring the tensorflow network graph.

## Instructions on Running 
Download the entire zip folder of our repo and run the jupyter notebooks. 

## Data Locations in Google Cloud Bucket: 

Google Cloud Bucket:
https://console.cloud.google.com/storage/browser/inpainting-final-project 

- `/inpainting-final-project/images/CelebA/img_align_celeba`
- `/inpainting-final-project/images/Cars/cars_test/cars_test`
- `/inpainting-final-project/images/Cars/cars_train`
- `/inpainting-final-project/images/SVHN`

Guide to access Cloud Bucket using Python

Prerequisite: 
1. ```pip install google-cloud-storage```
2. Follow the doc to Create service account key and add it to the bucket permission members
https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
3. Run export `GOOGLE_APPLICATION_CREDENTIALS="path_to_your_key_json_file"` in the shell; it only exists 


## Original Data Location 
CelebA dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 

Stanford Car dataset
https://ai.stanford.edu/~jkrause/cars/car_dataset.html 

Street View House Numbers (SVHN) dataset 
http://ufldl.stanford.edu/housenumbers/ 

