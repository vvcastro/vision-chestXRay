from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.measure import moments_central, moments_normalized
from skimage import filters, morphology
from scipy import stats
import numpy as np
import cv2


def compute_binary_img(image):
    """ Computes the binary [0 and 1] of the image, showing the contours and closing some imperfections. """
    binary = image > filters.threshold_otsu(image)
    clean = morphology.remove_small_objects(
        binary, binary.size // 100.0, connectivity=2)
    closed = morphology.binary_closing(clean)
    region = morphology.remove_small_holes(
        closed, binary.size // 100.0, connectivity=2).astype(np.float32)
    return cv2.normalize(region, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def extract_hu_moments(image):
    """ Calculate Hu Moments """
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments[:-3]

def extract_fourier_descriptors(image, num_descriptors=30):
    """ Extract features from the Fourier transform of the image """
    bin_img = compute_binary_img(image)
    contours, _ = cv2.findContours( bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_values = num_descriptors * 2

    if len(contours) > 0:
        contour = contours[0]
        contour = contour[:, 0, :]
        fourier_coeffs = np.fft.fft(contour, axis=0)[:num_descriptors]
        output = fourier_coeffs.flatten()
        if ( len(output) < total_values ):
            diff = total_values - len(output)
            return np.concatenate([output, np.full(diff, np.mean(output))])
    else:
        output = np.empty(total_values)
        output.fill(np.nan)
    return output


def extract_gabor_features(image, num_orientations=8, num_scales=4, num_lambdas=4):
    """ Calculate features from applying Gabor filters to the image """
    ksize = 3 if image.shape[0] <= 256 else 5

    # create combinations of rotations and scales
    gabor_features = []
    for theta in np.arange(start=0, stop=np.pi, step=(np.pi / num_orientations)):
        for sigma in range(1, num_scales + 1):
            for wave in range(1, num_lambdas + 1):
                wavelength = 5 * wave

                # get the gabor kernel from cv2 and convolve it with the image
                gabor_kernel = cv2.getGaborKernel( (ksize, ksize), sigma, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F)
                gabor_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
                gabor_features.append(gabor_image.mean())
    return np.array(gabor_features)


def extract_basic_intensity_statistics(image):
    """ Perform some basic pixel intensity statistics """
    mean = np.mean(image)
    std = np.std(image)
    kurtosis = stats.kurtosis(image.flatten())
    skew = stats.skew(image.flatten())
    return np.array([mean, std, kurtosis, skew])


def extract_local_binary_patterns(image, num_points=24, radius=3):
    """ Computes the LBP image and then extracts the relevant histograms from it."""
    lbp_image = local_binary_pattern( image, num_points, radius, method="uniform")
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange( 0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_histogram_of_oriented_gradients(image, num_bins=9):
    """ It selects a cell size depending on the image_size and compute the HOG"""
    ksize = 24 if image.shape[0] <= 256 else 32
    hog_features = hog(image, orientations=num_bins, pixels_per_cell=(ksize, ksize), cells_per_block=(1, 1))
    return hog_features


def extract_haralick_features(image, num_angles=8):
    """ Calculates the gray co-occurence matrix of the image. From the co-matrix, calculates the
        relevant features.
    """
    angles = [i * (np.pi / num_angles) for i in range(num_angles)]
    graycomat = graycomatrix(image, distances=[1], angles=angles)

    haralick_features = []
    for prop_name in ["contrast", "dissimilarity"]:
        feature = graycoprops(graycomat, prop_name).ravel()
        haralick_features.extend(feature)
    return np.array(haralick_features)

functions = {
    'hu_bin': lambda img: extract_hu_moments(compute_binary_img(img)),
    'fourier': extract_fourier_descriptors,
    'gabor': extract_gabor_features,
    'stats': extract_basic_intensity_statistics,
    'lbp': extract_local_binary_patterns,
    'hog': extract_histogram_of_oriented_gradients,
    'haralick': extract_haralick_features,
}


class DataFeatures:

    def __init__(self, features):

        # define feature extractions
        self.extractors = lambda img: { ft: functions[ft](img, **params) for ft, params in features }

        # group features
        self.geo_features = ['hu_bin', 'fourier_real', 'fourier_imag']
        self.int_features = ['gabor', 'stats']
        self.tex_features = ['lbp', 'hog', 'haralick']
        self.all_features = self.geo_features + self.int_features + self.tex_features

    def extract_features( self, image ):
        """ Extract all the defined features from the relevant image """
        extracted_features = self.extractors(image)
        extracted_features['fourier_real'] = np.real( extracted_features['fourier'] )
        extracted_features['fourier_imag'] = np.imag( extracted_features['fourier'] )
        extracted_features.pop('fourier')

        return extracted_features
