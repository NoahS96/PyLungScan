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
Contains a function to read .dicom files from a directory into a 3D array. Also 
contains a function to display the array in pyplot but cannot currently display 
along the z axis.
