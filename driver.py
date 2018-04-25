#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import argparse
from DicomReader import DicomReader
from ImageMath import ImageMath
from CNeuralNetwork import CNeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--patients', '-p', type=str, help="Path to patient folders directory")
parser.add_argument('--resampled', '-r', type=str, help="Path to directory with preprocessed images")
parser.add_argument('--csv', '-c', type=str, help="Path to a csv file with patient id's and their diagnosis")
parser.add_argument('--downsize', '-d', nargs='?', default=50, type=int, help="Value to size downsize the patient images to")
parser.add_argument('--tslices', '-ts', nargs='?', default=20, type=int, help="How many slices the image should be resampled to")
parser.add_argument('--epochs', '-e', nargs='?', default=10, type=int, help="How many training epochs to use")
parser.add_argument('--saver', '-s', nargs='?', type=str, help="Path to save a tensor training model save file")
args = parser.parse_args()

patient_dir = args.patients 
resample_dir = args.resampled
patients_csv = args.csv
slice_count = args.tslices
downsize_shape = args.downsize
epochs = args.epochs
saver = None

patientPathArray = []       # Holds the paths to the patient image data directories
processPatientArray = []    # Holds the paths to the unprocessed patient image directories

# Check if windows to set path splitter
file_split = '/'
if os.name =='nt':
    file_split = '\\'

if args.saver is not None:
    saver = args.saver

    #Create the file if it doesn't exist
    if not os.path.isfile(saver):
        print('Creating %s' % (saver))
        fh = open(saver, 'w+')
        fh.close()


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
        if patient.split(file_split)[-1] + '.npy' not in files:
            processPatientArray.append(patient)

# Get diagnosis labels from patient csv
data = pd.read_csv(patients_csv)

# Dictionary of patients and their diagnosis: key(patient_id) : value(cancer_diagnosis)
patient_diagnosis = {}
print('Getting Patient Diagnosis From %s' % (patients_csv))
for index, row in data.iterrows():
    if row['cancer'] == 0:
        patient_diagnosis[row['id']] = np.array([1,0])
    else:
        patient_diagnosis[row['id']] = np.array([0,1])


# Resample the images not found in the resampled directory.
# This will take some time.
for i in range(len(processPatientArray)):
    patient_name = processPatientArray[i].split(file_split)[-1]
    print('[%d/%d]\tPatient %s' % (i+1, len(processPatientArray), patient_name)) 

    print('\tReading Slices...')
    slices = DicomReader.readFromDir(processPatientArray[i])

    print('\tConverting to hu...')
    image = DicomReader.convertHounsfield(slices)

    print('\tDownsizing...')
    resampled_image = ImageMath.downsize(image, downsize_shape)

    print('\tReshaping Slices...')
    resampled_image = ImageMath.chunkify(resampled_image, slice_target=slice_count)

    print('\tExtracting Lung Data...')
    lungs = DicomReader.segmentLungMask(resampled_image, False)

    print('\tNormalizing...')
    lungs = ImageMath.normalize(lungs)
    
    try:
        write_data = [lungs, patient_diagnosis[patient_name]]
        print('\tWriting to %s' % (resample_dir + file_split + patient_name + '.npy'))
        np.save(resample_dir + file_split + patient_name, write_data)
    except KeyError:
        print('Skipping: Diagnosis not found for %s' % (patient_name))
        patientPathArray.remove(processPatientArray[i])
        #if i+1 >= len(processPatientArray):
        #    break

# patient_generator
#   Parameters:
#       patient_array   -   array of paths to patient folders
#       resample_dir    -   directory of resampled images
#   Purpose:
#       Because the images are fairly large, we use a generator to pass them in as needed to 
#       the neural network.
def patient_generator(patient_array, resample_dir):
    for patient in patient_array:
        yield np.load('./' + resample_dir + patient.split(file_split)[-1] + '.npy')

# Running the CNN 
print('Running CNN Now...')
nn = CNeuralNetwork(downsize_shape, slice_count, saver_path=saver)
print('Total loss: %.2f' % (nn.train_neural_network(patient_generator(patientPathArray, resample_dir), epochs)))


