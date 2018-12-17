### Image_inpainting_with_DGM
Final project for the course of Deep Learning at Columbia University. 

Paper:  https://arxiv.org/pdf/1607.07539.pdf
Readme:


Google Cloud Bucket:
https://console.cloud.google.com/storage/browser/inpainting-final-project 

Data locations in  Google Cloud Bucket:

/inpainting-final-project/images/CelebA/img_align_celeba
/inpainting-final-project/images/Cars/cars_test/cars_test
/inpainting-final-project/images/Cars/cars_train
/inpainting-final-project/images/SVHN

Guide to access Cloud Bucket using Python

Prerequisite: 
pip install google-cloud-storage
Follow the doc to Create service account key and add it to the bucket permission members
Run export GOOGLE_APPLICATION_CREDENTIALS="path_to_your_key_json_file" in the shell; it only exists 


https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python

CelebA dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html --

Stanford Car dataset
https://ai.stanford.edu/~jkrause/cars/car_dataset.html -- 

Street View House Numbers (SVHN) dataset 
http://ufldl.stanford.edu/housenumbers/ -- 


The code should be accompanied by a detailed readme file describing the following: instructions how to run the code, the name of the main jupyter notebook, function of each file in the project, where are the datasets, and other supporting information.
