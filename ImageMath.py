import numpy as np
import cv2

class ImageMath:

    MAX_AXIS = 150
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    #chunks
    #   Parameters:
    #       image   -   A numpy array   
    #       size    -   The size of chunks to return
    def chunks(image, size):
        for i in range(0, len(image), n):
            yield image[i:i + n]

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

    # normalize
    #   Parameters:
    #       image   -   A 3D dicom image
    #   Purpose:
    #       Remove all values above the MAX_BOUND value
    def normalize(image, min_bound=MIN_BOUND, max_bound=MAX_BOUND):
        image = (image - min_bound) / (max_bound - min_bound)
        image[image > 1] = 1.0
        image[image < 0] = 0.0
        return image

    # downsize
    #   Parameters:
    #       slices  -   Patient Dicom slices
    #       size    -   The size to set the image to 
    def downsize(slices, size):
        new_slices = []
        for num, each_slice in enumerate(slices):
            piece = cv2.resize(np.array(each_slice), (size, size))
            new_slices.append(piece)
        return np.array(new_slices)

