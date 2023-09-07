# In this script we extract and store the
# features for each of the data-splits we created

# The features parameters are the ones we discussed
# before on the `02_features_exploration` notebook.
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import sys
import os

# add the base of the project for imports
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_DIR = 'data'

from utils.features import DataFeatures

features = [
    ('hu_bin', {}), ('stats', {}),
    ('fourier', {'num_descriptors': 32}),
    ('gabor', {'num_orientations': 8, 'num_scales': 3, 'num_lambdas': 4}), 
    ('lbp', {'num_points': 32, 'radius':3}),
    ('haralick', {'num_angles': 4})
]
features_extractor = DataFeatures( features )

# data groups
groups = [ 
    "small_without_noise_not_inverted", "small_with_noise_not_inverted",
    "small_without_noise_inverted", "small_with_noise_inverted",
    "medium_without_noise_not_inverted", "medium_with_noise_not_inverted",
    "medium_without_noise_inverted", "medium_with_noise_inverted",
]

# extract features
all_files = pd.read_csv( os.path.join(BASE_DATA_DIR, 'train', 'labels_train.csv'))
filenames = all_files['file'].values
for group_name in groups:
    group_base_dir = os.path.join( BASE_DATA_DIR, 'subsets', group_name )
    group_features = {}
    for filename in tqdm(filenames, desc=group_name):
        img_path = os.path.join( group_base_dir, filename )
        img = cv2.imread( img_path, cv2.IMREAD_GRAYSCALE )
        img_features = features_extractor.extract_features(img)
        group_features[filename] = img_features
    np.save(os.path.join(BASE_DATA_DIR, 'features', group_name), group_features)