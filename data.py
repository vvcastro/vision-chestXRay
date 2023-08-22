# In this file the main pre-processing and data loading
# classes are presented

import numpy as np
import cv2
import os

equalisator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))

def resize_image( image, image_size):
    resized_image = cv2.resize( image, dsize=image_size)
    return cv2.normalize( resized_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def invert_image( image ):
    return cv2.bitwise_not( image )

def remove_img_borders( image, thresh=10 ):
    """ The idea is simple, check rows and cols that are all 0 or 255"""
    
    # we calculate the std on the rows and on the columns
    row_std = np.std(image, axis=1)
    col_std = np.std(image, axis=0)
    
    # real rows/cols
    significant_rows = np.where(row_std > thresh)[0]
    significant_cols = np.where(col_std > thresh)[0]

    # final rectangle to return
    i = significant_rows[0] if significant_rows.size > 0 else image.shape[0]
    j = significant_rows[-1] + 1 if significant_rows.size > 0 else -1
    x = significant_cols[0] if significant_cols.size > 0 else image.shape[1]
    y = significant_cols[-1] + 1 if significant_cols.size > 0 else -1
    return image[i:j, x:y]

def apply_laplace_sharpening( image, kernel_size=3, factor=1):
    """ Apply the laplace filtering as discussed before """
    image = cv2.normalize( image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, ddepth=cv2.CV_32F)
    laplace_img = cv2.Laplacian(image, ddepth=cv2.CV_32F, ksize=kernel_size)
    corrected_img = np.clip( image - factor * laplace_img, a_min=0, a_max=255)
    return corrected_img

def remove_noise( image, kernel_size=(5, 5) ):
    """ Apply a gaussian blur and then correct with laplace sharpening """
    blurred_image = cv2.GaussianBlur( image, ksize=kernel_size, sigmaX=0)
    corrected_image = DataProcessor.apply_laplace_sharpening( blurred_image, kernel_size=3, factor=1.7)
    return corrected_image


class DataProcessor:

    def __init__( self, data_dir, settings ):
        self.data_dir = data_dir

        self.image_size = settings['image_size']
        self.noise_std_thresh = settings['noise_std_thresh']
        self.inv_mean_thresh = settings['inv_mean_thresh']
        pass

    def apply_preprocessing(self, image):
        pass

    def store_in_memory(self, output_dir):
        pass

    
