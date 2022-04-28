# American-Sign-Language-with-Mediapipe

Estimate hand pose using MediaPipe(Python version).
This is a sample program that recognizes hand signs and finger gestures with a simple transfer learning mobileNet using the detected key points.





## Prerequisites

Before you continue, ensure you meet the following requirements:

* You have installed Anaconda python 3.8 or later.
* You have installed mediapipe 0.8.1 and OpenCV 3.4.2 or later.
* You have installed Tensorflow 2.5.0 or later.
* If you haven't installed Tensorflow yet, you can install it from this [tuturial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation).

## Dataset
#### About the dataset
I used my own dataset, you can check and use it [here](https://drive.google.com/drive/folders/1Cd2qNzgq-ox2Iwu12F1IXPOYeyw34Cdb?usp=sharing). 
Or you can create your own dataset by running collect.py file (remember to change the saved image location in this file for your customization).
If you want to use your dataset, then you have to label all your image for transfer learning by using labelImage tool, you can see tuturial [here](https://github.com/tzutalin/labelImg.)
#### Hand-sign alphabet
This program covers 24/26 characters of Latin alphabet (except J and Z because it required motion process). I used image below to generate my dataset, 



## Training model
#### Directory
Create your project folder and create folders like I did below


