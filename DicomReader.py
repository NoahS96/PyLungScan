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

            # Slope is generally 1 but occassionally is not
            if slope != 1:

                # Multiply as a float for precision then convert back to int
                images[slice_index] = slope * images[slice_index].astype(np.float64)
                images[slice_index] = images[slice_index].astype(np.int16)

            images[slice_index] += np.int16(intercept)

        return np.array(images, dtype=np.int16)

    
    # resamplePixels
    #   Parameters:
    #       images  -   Images converted to Hounsfield unit pixels
    #       slices  -   Array of Dicom images
    #   Purpose:
    #       When feeding the different patient images into a NN, the CT images may have different
    #       pixel spacings. The images need to have a standardized pixel spacing to be effective 
    #       in the neural network. Resamples the pixel spacing to [1,1,1].
    def resamplePixels(images, slices, new_spacing=[1,1,1]):
        cur_pixel_spacing = np.array(scan[0].PixelSpacing)
        spacing = np.array([slices[0].SliceThickness] + cur_pixel_spacing, dtype=np.float32)

        #Currently has a shape error
        resizing_factor = spacing/new_spacing
        new_real_shape = image.shape * resizing_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape/image.shape
        new_spacing = spacing/real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing

