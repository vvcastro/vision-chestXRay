#################
# In this file we have the preprocessing and storing of the images.
#################
from tqdm import tqdm
import pandas as pd
import sys
import os

# add the base of the project for imports
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGINAL_DATA_DIR = os.path.join( 'data', 'train' )

from utils.preprocessing import DataProcessor

# There are different settings to explore, here we are setting
# 8 relevant 
sizes = {
    'small': (128, 128),
    'medium': (224, 224),
}

cancel_noise = {
    'without_noise': True,
    'with_noise': False,
}

to_invert = {
    'inverted': True,
    'not_inverted': False,
}

settings = []
for img_size in sizes:
    for noise_type in cancel_noise:
            for invert_type in to_invert:
                this_settings = {
                    'id': f"{img_size}_{noise_type}_{invert_type}",
                    'image_size': sizes[img_size],
                    'noise': cancel_noise[noise_type],
                    'invert': to_invert[invert_type],
                    'sharpening': True
                }
                settings.append( this_settings )


###########################
# 1. Apply the preprocessing over all the dataset
# -> Note the values were calculated only on the training set

# load data.csv
all_files = pd.read_csv( os.path.join(ORIGINAL_DATA_DIR, 'labels_train.csv'))
filenames = all_files['file'].values

for config in settings:
    processor = DataProcessor(data_dir=ORIGINAL_DATA_DIR, settings=config)

    # preprocess images from the data dir
    for filename in tqdm( filenames, desc=config['id'] ):
        img = processor.read_image( filename )
        processed_img = processor.apply_preprocessing( img )
        processor.store_in_memory( filename, processed_img )
