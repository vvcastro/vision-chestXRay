# In this file the main pre-processing and data loading
# classes are presented

import numpy as np
import cv2
import os

equalisator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))

def resize_image( image, size, norm_type=cv2.NORM_MINMAX ):
    image = cv2.resize( image, dsize=size )
    return cv2.normalize( image, dst=None, alpha=0, beta=255, norm_type=norm_type )

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

def remove_noise( image, kernel_size ):
    """ Apply a gaussian blur and then correct with laplace sharpening """
    blurred_image = cv2.GaussianBlur( image, ksize=kernel_size, sigmaX=0)
    corrected_image = laplace_sharpening( blurred_image, kernel_size=3, factor=0.5)
    return corrected_image

def invert_image( image ):
    return cv2.bitwise_not( image )

def laplace_sharpening( image, kernel_size=3, factor=1):
    """ Apply the laplace filtering as discussed before """
    norm_type = cv2.NORM_MINMAX
    image = cv2.normalize( image, None, 0, 255, norm_type=norm_type, dtype=cv2.CV_32F)
    laplace_img = cv2.Laplacian(image, ddepth=cv2.CV_32F, ksize=kernel_size)
    corrected_img = np.clip( image - factor * laplace_img, a_min=0, a_max=255)
    return corrected_img

class DataProcessor:

    def __init__( self, data_dir, settings ):
        self.data_dir = data_dir

        self.image_size = settings['image_size']
        self.clean_noise = settings['noise']
        self.invert_images = settings['invert']
        self.sharpen_images = settings['sharpening']

        self.define_thresholds()

    def define_thresholds(self, noisy=53, inverted=65, blurred=1000):
        """ Define some basic settings used in the pipeline """
        self.noise_thresh = noisy
        self.inverted_thresh = inverted
        self.blurred_thresh = blurred

        # kernel size ( they depend on image size )
        img_dim = self.image_size[0]
        self.noise_ksize = (3, 3) if img_dim <= 256 else (7, 7)

    def read_image( self, image_name: str):
        """ Read the image and resize is to the set size. """
        image_path = os.path.join(self.data_dir, image_name)
        
        # read and resize to wanted size
        img = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE )
        img = resize_image( img, size=self.image_size)
        return img
    
    def apply_preprocessing(self, image, output_type=cv2.CV_8UC1):
        """ Applies the pre-processing steps on the `01_preprocessing` notebok. """
        norm_type = cv2.NORM_MINMAX

        # remove borders
        image = remove_img_borders( image )
        image = resize_image( image, size=self.image_size, norm_type = cv2.NORM_MINMAX )
        image = equalisator.apply( image )
        
        # (1) Check if it is a noisy image
        if ( self.invert_images ):
            img_crops = self.extract_relevant_patches( image )
            noise_metric = np.mean([ crop.std() for crop in img_crops ])
            if ( noise_metric >= self.noise_thresh ):
                image = remove_noise( image, kernel_size=self.noise_ksize )
                image = cv2.normalize( image, None, 0, 255, norm_type=norm_type, dtype=output_type )
                return  equalisator.apply( image )

        # (2) Check if inverted and blurred
        if ( self.invert_images ):
            img_crops = self.extract_relevant_patches( image )
            inv_metric = np.mean([ img_crops[2].mean(), img_crops[3].mean()])
            inv_metric -= 0.5 * np.mean([ img_crops[0].mean(), img_crops[1].mean()])
            if ( inv_metric <= self.inverted_thresh ):
                image = invert_image( image )
        
        # (3) Check blurred
        if ( self.sharpen_images ):
            blurred_metric = np.log( cv2.Laplacian( image, ddepth=-1 ).var() )
            sharpening_factor = 2 - blurred_metric * ( 2 / 3 )
            if ( sharpening_factor > 0 ):
                image = laplace_sharpening( image, factor=sharpening_factor)
    
        # normalise and equlise the histogram
        image = cv2.normalize( image, None, 0, 255, norm_type=norm_type, dtype=output_type )
        image = equalisator.apply( image )
        return image

    def extract_relevant_patches(self, image):
        """ 
        Takes from the image all the relevant patches (corners) and center crops.
        """
        crops = []

        img_center_y, img_center_x = image.shape[0] // 2, image.shape[1] // 2

        # take top corners
        crop_height, crop_width = 32, 32
        crops.append( image[ :crop_height, :crop_width] ) # corner0_crop
        crops.append( image[ :crop_height, -crop_width:] ) # corner1_crop

        # center crop
        crops.append( image[ 
            img_center_y - (crop_height//2):img_center_y + (crop_height//2),
            img_center_x - (crop_width//2):img_center_y + (crop_width//2)
        ] )

        # take mid bottom crop
        crop_height, crop_width = 64, 128
        crops.append( image[
            -crop_height:,
            img_center_x - (crop_width // 2):img_center_y + (crop_width // 2)
        ] )
        return crops


    def store_in_memory(self, output_dir):
        pass

    
