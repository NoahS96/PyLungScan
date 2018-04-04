#!/usr/bin/env python3

import pydicom as dicom
import os
import numpy as np
from matplotlib import pyplot as plt

#dirPath = './Data/LungCT-Diagnosis/R_004'

class DicomReader:

    #   Parameters:
    #       dirPath -   A path to the directory or parent directory of the .dicom files
    #   Purpose:
    #       Return an array of Dicom files in sequence
    def readFromDir(dirPath):
        slices = []
        
        # read each dicom file into the slices array then sort by position
        for dirName, subdirList, fileList in os.walk(dirPath):
            for filename in fileList:
                if '.dcm' in filename.lower():
                    slices.append(dicom.read_file(os.path.join(dirName, filename)))
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        # get the distance between slices 
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        # set the slice thickness of each slice
        for s in slices:
            s.SliceThickness = slice_thickness

        return slices

    # covertHounsfield
    #   Parameters:
    #       slices  -   An array of dicom data from one patient
    #   Purpose:
    #       Convert the pixel data to Hounsfield Units which is the standard measurement of 
    #       radiodensity which varies by tissue type. Because CT images are usually stored with
    #       unsinged ints, the pixels must be converted back to Hounsfield units which can have
    #       negative values. By converting to HU we can find specific tissue types from the images.
    def convertHounsfield(slices):
        images = np.stack([s.pixel_array for s in slices])
        images = images.astype(np.int16)

        # pixels of value -2000 represent pixels outside of the scan. Needs to be set to 0 for normalization.
        images[images == -2000] = 0

        for slice_index in range(len(slices)):

            # Get the rescale intercept from the Dicom file for conversion with a linear function. 
            intercept = slices[slice_index].RescaleIntercept
            slope = slices[slice_index].RescaleSlope

            if slope != 1:

                # Multiply as a float for precision then convert back to int
                images[slice_index] = slope * images[slice_index].astype(np.float64)
                images[slice_index] = images[slice_index].astype(np.int16)

                images[slice_index] += np.int16(intercept)

        return np.array(images, dtype=np.int16)


        

