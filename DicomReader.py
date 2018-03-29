#!/usr/bin/env python3

import pydicom as dicom
import os
import numpy as np
from matplotlib import pyplot as plt

#dirPath = './Data/LungCT-Diagnosis/R_004'

class DicomReader:

    # readFromDir
    #   Parameters:
    #       dirPath -   A path to the directory or parent directory of the .dicom files
    #   Purpose:
    #       Return a 3-dimensional array of the .dicom files in the provided directory
    def readFromDir(dirPath):
        filesDCM = []
        for dirName, subdirList, fileList, in os.walk(dirPath):
            for filename in fileList:
                if '.dcm' in filename.lower():
                    filesDCM.append(os.path.join(dirName, filename))


        # Use the first dicom file to get metadata
        RefDicom = dicom.read_file(filesDCM[0])
        ConstPixelDims = (int(RefDicom.Rows), int(RefDicom.Columns), len(filesDCM))
        ConstPixelSpacing = (float(RefDicom.PixelSpacing[0]), float(RefDicom.PixelSpacing[1]), float(RefDicom.SliceThickness))

        # Setup the arrays for each dimension of the image
        x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
        y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
        z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

        DicomArray = np.zeros(ConstPixelDims, dtype=RefDicom.pixel_array.dtype)

        # Add the pixel data to the dicom array
        for filenameDCM in filesDCM:
            ds = dicom.read_file(filenameDCM)
            DicomArray[:, :, filesDCM.index(filenameDCM)] = ds.pixel_array
        
        return DicomArray
        
    # show
    #   Parameters:
    #       DicomArray  -   A 3-dimensional array of .dicom data
    #   Purpose:
    #       Display the .dicom data on a pyplot graph to determine if the array is correct
    def show(DicomArray):
        plt.figure(dpi=300)
        plt.axes().set_aspect('equal', 'datalim')
        plt.set_cmap(plt.gray())
        plt.pcolormesh(x, y, np.flipud(DicomArray[:, :, 67]))
        plt.show()
