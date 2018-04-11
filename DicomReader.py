#!/usr/bin/env python3

import pydicom as dicom
import scipy.ndimage
import os
import numpy as np
from matplotlib import pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DicomReader:

    MAX_AXIS = 500
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

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
        slice_thickness = 0
        i = 0
        try:
            # Loop needed to make sure thickness is greater than 0 because some scans might have 
            # slices with the same position.
            while slice_thickness == 0 and i+1 < len(slices):
                slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePositionPatient[2])
                i += 1
        except:
            while slice_thickness == 0 and i+1 < len(slices):
                slice_thickness = np.abs(slices[i].SliceLocation - slices[i+1].SliceLocation)
                i += 1

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
    #       in the neural network. Resamples the pixel spacing to [1,1,1]. This function takes a 
    #       while to complete due to the scipy interpolation.
    def resamplePixels(image, slices, new_spacing=[1,1,1]):
        
        # Create an array that represents the pixel spacing of x,y,z
        cur_pixel_spacing = np.array(slices[0].PixelSpacing, dtype=np.float64)
        cur_slice_thickness = np.array(slices[0].SliceThickness, dtype=np.float64)
        spacing = np.append(cur_slice_thickness, cur_pixel_spacing)

        # Calculate the new resizing factor
        resizing_factor = spacing/new_spacing
        new_real_shape = image.shape * resizing_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape/image.shape
        new_spacing = spacing/real_resize_factor

        # Resize
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing

    # largestLabelVolume
    #   Parameters:
    #       image   -   Pixel array of patient
    #       bg      -   Value to search for in array
    #   Purpose:
    #       Determine the largest label of air around the patient. This will be kept in the patient
    #       image.
    def largestLabelVolume(image, bg=-1):
        vals, counts = np.unique(image, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    # segmentLungMask
    #   Parameters:
    #       image   -   Pixel array of patient
    #       fill_lung_structures    -   Boolean value determining whether lungs in patient image
    #           should be hollow or filled
    #   Purpose:
    #       Get a mask (i.e. image) of the lungs from the patient image.
    def segmentLungMask(image, fill_lung_structures=True):

        # -320 is the threshold determining which pixels are used
        binary_image = np.array(image > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)
        
        # Get the label of the air background
        background_label = labels[0,0,0]
        binary_image[background_label == labels] = 2

        if fill_lung_structures:
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = DicomReader.largestLabelVolume(labeling, bg=0)

                if  l_max is not None:
                    binary_image[i][labeling != l_max] = 1
        
        # Make the image binary then invert the lung pixels
        binary_image -= 1
        binary_image = 1-binary_image

        # Remove extra air pockets
        labels = measure.label(binary_image, background=0)
        l_max = DicomReader.largestLabelVolume(labels, bg=0)
        if l_max is not None:
            binary_image[labels != l_max] = 0

        return binary_image


    # normalize
    #   Parameters:
    #       image   -   A 3D dicom image
    #   Purpose:
    #       Remove all values above the MAX_BOUND value
    def normalize(image):
       image = (image - DicomReader.MIN_BOUND) / (DicomReader.MAX_BOUND - DicomReader.MIN_BOUND)
       image[image > 1] = 1.0
       image[image < 0] = 0.0
       return image

    # pad_width
    #   Parameters:
    #       image   -   A 3D image numpy array
    #       value   -   The value to pad the image with
    #       size    -   The cubic size to set the image
    #   Purpose:
    #       Pad the images so that they are all of the same shape.
    def pad_with(image, value, size=MAX_AXIS):
        shape = image.shape
        x_value = int((size-shape[0])/2)
        y_value = int((size-shape[1])/2)
        z_value = int((size-shape[2])/2)

        if (size-shape[0])%2 == 0:
            x_set = (x_value, x_value)
        else:
            x_set = (x_value, x_value+1)

        if (size-shape[1])%2 == 0:
            y_set = (y_value, y_value)
        else:
            y_set = (y_value, y_value+1)

        if (size-shape[2])%2 == 0:
            z_set = (z_value, z_value)
        else:
            z_set = (z_value, z_value+1)


        #Pad the image so the main image is in the center with a padding border of 0's
        npad = (x_set, y_set, z_set)
        new_image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
        
        return new_image
