# ---- This is <classify_S1_image_from_feature_folder.py> ----

"""
Example for easy use of 'ice_type_classification' library for S1 classifiation.
"""

import pathlib

import ice_type_classification.classification as cl

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# define the basic data directory for your project
BASE_DIR = pathlib.Path('~/Data/examples/Sentinel-1/')

# define S1 name
S1_name = 'S1A_EW_GRDM_1SDH_20220502T074527_20220502T074631_043029_05233F_7BC7'

# build the path to input feature folder
feat_folder = BASE_DIR / f'features' / 'ML_3x3' / f'{S1_name}'

# build the path to output result folder
result_folder = BASE_DIR / 'results' / 'ML_3x3'

# choose classifier model
classifier_model_path = '/home/jo/work/ice_type_classification/src/ice_type_classification/clf_models/belgica_bank_ice_types.pickle'

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

cl.classify_S1_image_from_feature_folder(
    feat_folder,
    result_folder,
    classifier_model_path,
    valid_mask = False,
    block_size = 1000000.0,
    overwrite = False,
    loglevel = 'INFO',
)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classify_S1_image_from_feature_folder.py> ----

