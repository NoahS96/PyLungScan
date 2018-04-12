<img src="https://thumb7.shutterstock.com/display_pic_with_logo/167811904/767060047/stock-photo-portrait-of-doctor-in-blue-uniform-with-his-thumbs-up-isolated-over-white-background-people-767060047.jpg" align="right"/>

# PyLungScan
This project uses a neural network created with Tensorflow to analyze lung CT 
scan images in DICOM format and predict whether the images contain a cancerous tumour.
We hope to achieve the following goals:
 * Read in CT image datasets from DICOM files into Python
 * Create and train a neural network based on a lung cancer diagnosed dataset
 * Differentiate healthy lung images from cancerous ones
 * Achieve at least a 60 percent accuracy in diagnosis
 * Keep false positive occurances below 30 percent

## Issues

## DicomReader.py
Contains a function to read .dicom files from a directory into a 3D array. Added 
several other functions that convert the pixel data to hounsfield units and extracts lung
tissue from the images for better NN processing. 
Credit for said functions goes to Guido Zuidhof https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

## driver.py
Is passed in a root directory containing patient info in each of their respective 
subdirectories and a path to a directory containing presampled images. Passes these 
patient folders to the DicomReader and trains the neural network with the returned 
dicom array. Once sufficiently trained, the driver tests the neural network with 
random patient info and prints the predicted result and the expected result.

## ImageMath.py
Handles downsizing, chunking, and normalization of image arrays.

## CNeuralNetwork.py
Sets up the neural network with tensorflow and provides a method to train and test the 
network. Also provides a function to return the loss array.
