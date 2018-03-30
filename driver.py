import os
import numpy as np
from matplotlib import pyplot as plt

##########################
#       PSEUDOCODE       #
##########################
#
# from DicomReader import DicomReader
# from CNeuralNetwork import CNeuralNetwork
#
# rootDir = argv[0]
# patientArr = None
#
# //Get an array of the patient directories from the data directory
# //Directory structure is as follows ./LungCT-Diagnosis/R_0NN for some number NN
# for dir, subdir, file in os.walk('./LungCT-Diagnosis'):
#       patientArr = subdir
#       break       
#
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
