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

<p>your_directory/<p>
  <p>└─ TensorFlow/<p>
    <p>├─ addons/ (Optional)<p>
    <p>│    └─ labelImg/<p>
    <p>├─ models/ (adding by install tensorflow object detection API)<p>
    <p>│    ├─ community/<p>
    <p>│    ├─ official/<p>
    <p>│    ├─ orbit/<p>
    <p>│    ├─ research/<p>
    <p>│    └─ ...<p>
    <p>└─ workspace/<p>
workspace folder shall be our training folder, which will contain all files related to our model training. It is advisable to create a separate training folder each time we wish to train on a different dataset. The typical structure for training folders is shown below.
<p>workspace/<p>
  ├─ annotations/
  ├─ images/
  │  ├─ test/
  │  └─ train/
  ├─ models/
  └─pre-trained-models/

Here’s an explanation for each of the folders/filer shown in the above tree:

annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.

exported-models: This folder will be used to store exported versions of our trained model(s).

images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.

images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.

images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.

models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.

pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.


