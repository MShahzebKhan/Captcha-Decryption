# Captcha-Decryption
A simple convolutional neural network code for the decryption of captcha images. 
The network consist of three convolution layers followed by ReLU layers.

Required Software:

1- Python3*

2- Tensorflow* 

3- all the libraries import in both accuracy.py and test.py code

* latest version of the software required

test.py :
This file will ask to give input, i.e. the image name in integer e.g. 345 , and will save the predicted captcha in a folder name as Test_images in a text whose name is going to be same as the image name.

accuracy.py :
This file will calculate the accuracy of the algorithm and log the data in a file named as accuracy.txt. It will take input an integer value as a total no of images on which you want to calculate accuracy. 
On 1000 test images the accuracy is 83%.

There are three folders in the directory:

1- Test_images  ==> containing text file of the predicted captchas

2- Checkpoint   ==> it contains the learned parameters of the model 

3- Captcha      ==> it contains all the captcha images 


Other files in the directory:

1- content_format_02.txt  ==> it contains label of all the captcha images

2- accuracy.png           ==> A picture showing accuracy on 1000 images with few samples.

3- accuracy.txt           ==> It contains log data of the accuracy.py 


