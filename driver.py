import os
import re
import numpy as np
import os, re
from DicomReader import DicomReader


# still need to add the command line argument reader
# rootDir = argv[0]
patient_dir = './LungCT-Diagnosis'
resample_dir = './ResampledImages'

patientPathArray = []       # Holds the paths to the patient image data directories
processPatientArray = []    # Holds the paths to the unprocessed patient image directories


# Get an array of the patient directories from the data directory
# Directory structure is as follows ./LungCT-Diagnosis/R_0NN for some number NN
# Store the path and patient names in separate arrays
print('Gathering patient image directory %s' % (patient_dir))
regex = re.compile('.*R_\d\d\d$')
for dir_name, subdir, files in os.walk(patient_dir):
    if re.match(regex, dir_name):
        patientPathArray.append(dir_name)


# Check the ResampledImages directory for patient images that have already been 
# processed. If not, add them to a processing list.
print('Checking for unprocessed patient images in %s' % (resample_dir))
regex = re.compile('.*R_\d\d\d\.npy$')
for dir_name, subdir, files in os.walk(resample_dir):
    for patient in patientPathArray:
        if patient.split('/')[-1] + '.npy' not in files:
            processPatientArray.append(patient)


# Resample the images not found in the resampled directory.
# This will take some time.
for i in range(len(processPatientArray)):
    patient_name = patient.split('/')[-1]
    print('[%d/%d] Patient %s' % (i+1, len(processPatientArray), patient_name)) 

    print('\tReading Slices...')
    slices = DicomReader.readFromDir(patient)

    print('\tConverting to hu...')
    image = DicomReader.convertHounsfield(slices)

    print('\tResampling...')
    resampled_image, new_spacing = DicomReader.resamplePixels(image, slices)

    print('\tExtracting Lung Data...')
    lungs = DicomReader.segmentLungMask(resampled_image, False)
    np.save(resample_dir + '/' + patient_name, resampled_image)


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
