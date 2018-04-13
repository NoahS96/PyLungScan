#!/usr/bin/env python3

import os
import numpy as np
import argparse
from DicomReader import DicomReader
from ImageMath import ImageMath


parser = argparse.ArgumentParser()
parser.add_argument('--patients', '-p', type=str, help="Path to patient folders directory")
parser.add_argument('--resampled', '-r', type=str, help="Path to directory with preprocessed images")
parser.add_argument('--downsize', '-d', nargs='?', default=150, type=int, help="Value to size downsize the patient images to")
parser.add_argument('--tslices', '-ts', nargs='?', default=50, type=int, help="How many slices the image should be resampled to")
args = parser.parse_args()

patient_dir = args.patients 
resample_dir = args.resampled
slice_count = args.tslices
downsize_shape = args.downsize

patientPathArray = []       # Holds the paths to the patient image data directories
processPatientArray = []    # Holds the paths to the unprocessed patient image directories


# Get an array of the patient directories from the data directory
# Directory structure is as follows ./LungCT-Diagnosis/R_0NN for some number NN
# Store the path and patient names in separate arrays
print('Gathering patient image directory %s' % (patient_dir))
for dir_name, subdir, files in os.walk(patient_dir):
    if dir_name != patient_dir:
        print(dir_name)
        patientPathArray.append(dir_name)


# Check the ResampledImages directory for patient images that have already been 
# processed. If not, add them to a processing list.
print('Checking for unprocessed patient images in %s' % (resample_dir))
for dir_name, subdir, files in os.walk(resample_dir):
    for patient in patientPathArray:
        if patient.split('/')[-1] + '.npy' not in files:
            processPatientArray.append(patient)


# Resample the images not found in the resampled directory.
# This will take some time.
for i in range(len(processPatientArray)):
    patient_name = processPatientArray[i].split('/')[-1]
    print('[%d/%d]\tPatient %s' % (i+1, len(processPatientArray), patient_name)) 

    print('\tReading Slices...')
    slices = DicomReader.readFromDir(processPatientArray[i])

    print('\tConverting to hu...')
    image = DicomReader.convertHounsfield(slices)

    #print('\tResampling...')
    #resampled_image, new_spacing = DicomReader.resamplePixels(image, slices)
    
    print('\tDownsizing...')
    resampled_image = ImageMath.downsize(image, downsize_shape)

    print('\tReshaping Slices...')
    resampled_image = ImageMath.chunkify(resampled_image, slice_target=slice_count)

    print('\tExtracting Lung Data...')
    lungs = DicomReader.segmentLungMask(resampled_image, False)

    #print('\tAdding Padding Border...')
    #lungs = DicomReader.pad_with(lungs, 0)

    print('\tNormalizing...')
    lungs = ImageMath.normalize(lungs)
    
    print('\tWriting to %s' % (resample_dir + '/' + patient_name + '.npy'))
    np.save(resample_dir + '/' + patient_name, lungs)

# Walk throught the resample directory again and train the neural network with
# the lung images.

############
#   TODO   #
############
# //For each patient, send the dicomArray to the CNN for training
# for i in range(0, training_epochs):
#       for patient in patientArr:
#           dicomArray = DicomReader(patient)
#           CNeuralNetwork.train(dicomArray)
#
# //For a random patient, send the dicomArray for a test diagnosis
# for i in range(0, testing_epochs):
#       randomIndex = np.random.randomint(0, len(patientArr))
#       dicomArray = DicomReader(patientArr[randomIndex])
#       result = CNeuralNetwork.diagnose(dicomArray)
#       print(result)
#
# //Get the loss array from the Neural Network and display with matplotlib
# loss = CNeuralNetwork.getLoss()
# plt.show(loss)
