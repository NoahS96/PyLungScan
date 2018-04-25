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

## Dependancies
 * Tensorflow 1.5.1
 * pydicom 1.0.2
 * numpy 1.13.3
 * pandas 0.20.3
 * opencv 3.3.1
 * scipy 1.0.1 

## How to use
Arguments:
 * --patients|-p : Path to a directory of patient folders containing dicom files
 * --resampled|-r : Path to a directory of images already processed by the driver
 * --csv|-c : Path to csv file containing patient ids (same id as in patients folder) and their diagnosis (1 for cancer or 0 for no cancer)
 * --downsize|-d : Optional argument specifying the desired shape of the processed images. Default: 150
 * --tslices|-ts : Optional argument specifying the desired slice count to resample the image to. Default:50
 * --saver|-s : Optional argument specifying the path to an existing tensorflow model or one to be created. First create a directory to save the file in then choose a name for the model file. Example: create directory TrainingModel the specify -s /TrainingModel/model.ckpt
 
 Example:
 ./driver.py -p SampleImages/ -r TestResampled/ -c stage1_labels.csv -s ./TrainingModel/model.ckpt -e 20
 
 Make sure the driver has the execute permission. Keep all patient folders under a single directory and keep their corresponding dicom files in their appropriate patient folder. Create a directory to store the .npy files of the processed images. Provide the -p and -r arguments and run. Currently, the driver only preprocessess the images.
 
 Datasets can be found at:
  * https://www.kaggle.com/c/data-science-bowl-2017/data
  * https://wiki.cancerimagingarchive.net/display/Public/LungCT-Diagnosis (Note: There is no diagnosis csv file for this set)

### DicomReader.py
Contains a function to read .dicom files from a directory into a 3D array. Added 
several other functions that convert the pixel data to hounsfield units and extracts lung
tissue from the images for better NN processing. 
Credit for said functions goes to Guido Zuidhof https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

### driver.py
Is passed in a root directory containing patient info in each of their respective 
subdirectories and a path to a directory containing presampled images. Passes these 
patient folders to the DicomReader and trains the neural network with the returned 
dicom array. Once sufficiently trained, the driver tests the neural network with 
random patient info and prints the predicted result and the expected result.

### ImageMath.py
Handles downsizing, chunking, and normalization of image arrays.

### CNeuralNetwork.py
Sets up the neural network with tensorflow and provides a method to train and test the 
network. Also provides a function to return the loss array.
