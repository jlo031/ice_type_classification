# ---- This is <classification.py> ----

"""
Module for forward classification of satellite images.
""" 
import os
os.chdir(f'C:/Users/{os.getlogin()}/OneDrive - UiT Office 365/Uncertainty/')
from S1_uncertainty_2023 import Mahalanobis_distance

import argparse
import sys
import pathlib
import shutil
import copy

from loguru import logger

import numpy as np

from osgeo import gdal
from osgeo import osr

import config_sea_ice_classification.config_sea_ice_classification as conf
import JLib.S1_product_functions as S1
import JLib.read_write_img as rwi
import JLib.gaussian_IA_classifier as gia

import feature_extraction.S1_features as S1_feat
import feature_extraction.general_features as gen_feat

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def classify_image(
  feat_folder,
  result_folder,
  classifier_path,
  valid=False,
  block_size=1e6,
  overwrite=False,
  loglevel='INFO',
):

  """Classify input image

  Parameters
  ----------
  feat_folder : path to input feature folder
  result_folder : path to result folder where labels file is placed
  classifier_path : path to pickle file with classifier dict
  valid : use valid mask
  block_size : number of pixels for block-wise processing (default=1e6)
  overwrite : overwrite existing files (default=False)
  loglevel : loglevel setting (default='INFO')
  """

  # remove default logger handler and add personal one
  logger.remove()
  logger.add(sys.stderr, level=loglevel)

  logger.info('Classifying ice types in input image')

  logger.debug(f'{locals()}')
  logger.debug(f'file location: {__file__}')

  # get directory where module is installed
  module_path = pathlib.Path(__file__).parent.parent
  logger.debug(f'module_path: {module_path}')

# -------------------------------------------------------------------------- #

  # convert folder strings to paths
  feat_folder     = pathlib.Path(feat_folder).expanduser().absolute()
  result_folder   = pathlib.Path(result_folder).expanduser().absolute()
  classifier_path = pathlib.Path(classifier_path).expanduser().absolute()

  # convert block_size string to integer
  block_size = int(block_size)

  logger.debug(f'feat_folder:     {feat_folder}')
  logger.debug(f'result_folder:   {result_folder}')
  logger.debug(f'classifier_path: {classifier_path}')

  if not feat_folder.is_dir():
    logger.error(f'Cannot find feat_folder: {feat_folder}')
    raise NotADirectoryError(f'Cannot find feat_folder: {feat_folder}')

  if not classifier_path.is_file():
    logger.error(f'Cannot find classifier_path: {classifier_path}')
    raise FileNotFoundError(f'Cannot find classifier_path: {classifier_path}')

  # get input basename from feat_folder
  f_base = feat_folder.stem

  # define output file names and paths
  result_basename = f_base + '_labels'
  result_path     = result_folder / f'{result_basename}.img'
  result_path_hdr = result_folder / f'{result_basename}.hdr'

  # logger
  logger.debug(f'result_path {result_path}')

  # check if outfile already exists
  if result_path.is_file() and not overwrite:
    logger.info('Output files already exist, use `-overwrite` to force')
    return
  elif result_path.is_file() and overwrite:
    logger.info('Removing existing output file and classifying again')
    os.remove(result_path.as_posix())
    os.remove(result_path_hdr.as_posix())

# -------------------------------------------------------------------------- #

  # GET BASIC CLASSIFIER INFO

  # load classifier dictionary
  classifier_dict = gia.read_classifier_dict_from_pickle(
    classifier_path.as_posix()
  )

  if not 'type' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `type` key')
    raise KeyError(f'classifier_dict does not contain `type` key')

  if not 'required_features' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `required_features` key')
    raise KeyError(f'classifier_dict does not contain `required_features` key')

  # get clf_type
  clf_type = classifier_dict['type']

  # get list of required features for classifier
  required_features = sorted(classifier_dict['required_features'])

# -------------------------------------------------------------------------- #

  # CHECK EXISTING AND REQUIRED FEATURES

  # get list of existing features in feat_folder
  existing_features = sorted(
    [f for f in os.listdir(feat_folder) if f.endswith('img')]
  )

  # check that all required_features exist
  for f in required_features:
    if f'{f}.img' not in existing_features:
      logger.error(f'Cannot find required feature: {f}')
      raise FileNotFoundError(f'Cannot find required feature: {f}')

  # get Nx and Ny from first required feature
  Nx, Ny = S1.get_img_dimensions(
    (feat_folder / f'{required_features[0]}.img').as_posix()
  )

  # check that all required features have same dimensions
  for f in required_features:
    Nx_current, Ny_current = S1.get_img_dimensions(
      (feat_folder / f'{f}.img').as_posix()
    )
    if not Nx == Nx_current or not Ny == Ny_current:
      logger.error(f'Image dimensions of required features do not match')
      raise ValueError(f'Image dimensions of required features do not match')

# -------------------------------------------------------------------------- #

  # CHECK VALID AND IA MASK IF REQUIRED

  if valid:
    logger.info('Using valid mask')

    # check that valid.img exists
    if 'valid.img' not in existing_features:
      logger.error('Cannot find valid mask: valid.img')
      logger.error(
        'Unset `-valid` flag, if you do not want to use a valid mask'
      )
      raise FileNotFoundError('Cannot find valid mask: valid.img')

    else:
      # get valid_mask dimensions
      Nx_valid, Ny_valid = S1.get_img_dimensions(
        (feat_folder / 'valid.img').as_posix()
      )
      # check that valid_mask dimensions match feature dimensions
      if not Nx == Nx_valid or not Ny == Ny_valid:
        logger.error(f'valid_mask dimensions do not match featured imensions')
        raise ValueError(
          f'valid_mask dimensions do not match featured imensions'
        )

      # get valid_mask data type
      valid_data_type_in = gdal.Open(
        (feat_folder / 'valid.img').as_posix(), gdal.GA_ReadOnly
      ).GetRasterBand(1).DataType
      logger.debug(f'valid_data_type_in: {valid_data_type_in}')
      # set valid_mask_data_type for memory mapping
      if valid_data_type_in == 1:
        valid_mask_data_type = np.uint8
      elif valid_data_type_in == 6:
        valid_mask_data_type = np.float32
      else:
        logger.error('Unknown valid_mask data type')
      logger.debug(f'valid_mask_data_type: {valid_mask_data_type}')

  else:
    logger.info('Not using valid mask, set `-valid` if wanted')


  if clf_type == 'gaussian_IA':
    logger.info('Classifier is using IA information')

    # check that IA.img exists
    if 'IA.img' not in existing_features:
      logger.error(f'Cannot find IA image: IA.img')
      raise FileNotFoundError(f'Cannot find IA image: IA.img')

    else:
      # get IA dimensions
      Nx_IA, Ny_IA = S1.get_img_dimensions(
        (feat_folder / 'IA.img').as_posix()
      )
      # check that IA dimensions match feature dimensions
      if not Nx == Nx_IA or not Ny == Ny_IA:
        logger.error(f'IA dimensions do not match featured imensions')
        raise ValueError(f'IA dimensions do not match featured imensions')

  else:
    logger.info('Classifier is not using IA information')

# -------------------------------------------------------------------------- #

  # BUILD CLASSIFIER OBJECT ACCORDING TO CLASSIFIER DICT

  # check classifier type
  # for gaussian or gaussian_IA:
  #   build the actual clf object from the saved parameters
  # for other types:
  #    set the clf object directly from classifier_dict
  #    import the necessary modules from sklearn

  if clf_type == 'gaussian_IA':
    logger.debug(f'clf_type: {clf_type}')
    logger.debug('classifier_dict only contains clf parameters')
    logger.debug('Setting up classifier object from these parameters')
    clf = gia.make_gaussian_IA_clf_object_from_params_dict(
      classifier_dict['gaussian_IA_params']
    )
    # Extract model parameters needed for Mahalnobis distance calculation
    mu_vec_allClasses  = classifier_dict['gaussian_IA_params']['mu']
    cov_mat_allClasses = classifier_dict['gaussian_IA_params']['Sigma']
    n_classes          = int(classifier_dict['gaussian_IA_params']['n_class'])
    IA_0               = classifier_dict['gaussian_IA_params']['IA_0']
    IA_slope           = classifier_dict['gaussian_IA_params']['b']

  elif clf_type =='gaussian':
    logger.debug(f'clf_valid_mask_data_typetype: {clf_type}')
    logger.debug('classifier_dict only contains clf parameters')
    logger.debug('Setting up classifier object from these parameters')
    clf = gia.make_gaussian_clf_object_from_params_dict(
      classifier_dict['gaussian_params']
    )
    # Extract model parameters needed for Mahalnobis distance calculation
    mu_vec_allClasses  = classifier_dict['gaussian_params']['mu']
    cov_mat_allClasses = classifier_dict['gaussian_params']['Sigma']
    n_classes          = int(classifier_dict['gaussian_params']['n_class'])

  else:
    ##clf = classifier_dict['clf_object']
    logger.error('Not implemented yet')
    raise NotImplementedError('Not implemented yet')

# -------------------------------------------------------------------------- #

  # get image dimensions and total number of pixels)
  shape  = (Ny, Nx)
  N      = Nx*Ny

  # initialize data dict and/or data list
  data_dict = dict()
  data_list = []

  # logger
  logger.info('Memory mapping required data')

  # memory map IA if required, set empty otherwise
  if clf_type=='gaussian_IA':
    logger.debug('Memory mapping IA')
    IA_mask = np.memmap(
      (feat_folder / 'IA.img').as_posix(), 
      dtype=np.float32, mode='r', shape=(N)
    ) #.byteswap()
  else:
    IA_mask = np.zeros(N).astype(int)

  # memory map valid mask if required, set valid mask to 1 otherwise
  if valid:
    logger.debug('Memory mapping valid_mask')
    valid_mask = np.memmap(
      (feat_folder / 'valid.img').as_posix(), 
      dtype=valid_mask_data_type, mode='r', shape=(N)
    )#.byteswap()
  else:
    logger.debug('Setting valid_mask to 1')
    valid_mask = np.ones(N).astype(int)

  for f in required_features:
# GLCM features currently require memory mapping withoug byteswap
# texture features require byteswap
# this should be fixed at some point
    logger.debug(f'Memory mapping feature: {f}')
    if 'GLCM' in f:
      data_dict[f] = np.memmap(
        f'{feat_folder.as_posix()}/{f}.img', 
        dtype=np.float32, mode='r', shape=(N)
      )
    else:
      data_dict[f] = np.memmap(
        f'{feat_folder.as_posix()}/{f}.img', 
        dtype=np.float32, mode='r', shape=(N)
      )#.byteswap()

# -------------------------------------------------------------------------- #

  # initialize labels and probabilities
  labels_img = np.zeros(N)
  probs_img  = np.empty((N,n_classes))
  probs_img.fill(np.nan)
  Mahal_img  = np.empty((N,n_classes))
  Mahal_img.fill(np.nan)

  # find number of blocks from block_size
  n_blocks   = int(np.ceil(N/block_size))

  # logger
  logger.info('Performing block-wise processing of memory-mapped data')
  logger.info(f'block-size: {block_size}')
  logger.info(f'n_blocks:   {n_blocks}')

  # for progress report at every 10%
  log_percs   = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
  perc_blocks = np.ceil(
    np.array(n_blocks)*log_percs
  ).astype(int)

# -------------------------------------------------------------------------- #

  # loop over all blocks
  for block in np.arange(n_blocks):

    # logger
    logger.debug(f'Processing block {block+1} of {n_blocks}')

    if block in perc_blocks:
      logger.info(
        f'..... {int(log_percs[np.where(perc_blocks==block)[0][0]]*100)}%'
      )

    # get idx of current block
    idx_start = block*block_size
    idx_end   = (block+1)*block_size

    # select IA and valid mask for current block
    IA_block    = IA_mask[idx_start:idx_end]
    valid_block = valid_mask[idx_start:idx_end]

    # select all features for current block (make sure order is correct)
    X_block_list = []    
    for key in required_features:
      X_block_list.append(
        data_dict[key][idx_start:idx_end]
      )

    # stack all selected features to array (N,n_feat)
    X_block= np.stack(X_block_list,1)   

    # select only valid part of block
    X_block_valid  = X_block[valid_block==1]
    IA_block_valid = IA_block[valid_block==1]
    
    # predict labels where valid==1
    if clf_type == 'gaussian_IA':
      labels_img[idx_start:idx_end][valid_block==1], probs_img[idx_start:idx_end][valid_block==1] = \
        clf.predict(X_block_valid, IA_block_valid)
                
      # Calculate Mahalanobis distance for each class incuding IA dependency
      Mahal_img[idx_start:idx_end][valid_block==1] = \
        Mahalanobis_distance(X_block_valid, mu_vec_allClasses, cov_mat_allClasses, 
                        IA_test = IA_block_valid, IA_0 = IA_0, IA_slope = IA_slope)
   
    elif clf_type == 'gaussian':
      labels_img[idx_start:idx_end][valid_block==1], probs_img[idx_start:idx_end][valid_block==1] = \
        clf.predict(X_block_valid)
        
      # Calculate Mahalanobis distance for each class 
      Mahal_img[idx_start:idx_end][valid_block==1] = \
         Mahalanobis_distance(X_block_valid, mu_vec_allClasses, cov_mat_allClasses)

    else:
      labels_img[idx_start:idx_end][valid_block==1], probs_img[idx_start:idx_end][valid_block==1] = \
        clf.predict(X_block_valid)
        
	# set labels to 0 where valid==0	
    labels_img[idx_start:idx_end][valid_block==0] = 0	
  
    logger.info('Finished classification')	
  
  # reshape to image geometry	
  labels_img = np.reshape(labels_img,shape)
# -------------------------------------------------------------------------- #

  # create result_folder if needed
  result_folder.mkdir(parents=True, exist_ok=True)

  # get drivers
  output = gdal.GetDriverByName('Envi').Create(
    result_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte
  )

  # write labels_img to band 1
  output.GetRasterBand(1).WriteArray(labels_img)

  # flush
  output.FlushCache()

  # logger
  logger.info(f'Result writtten to {result_path}')

  return probs_img, Mahal_img, shape

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def inspect_classifier_pickle(
  classifier_path,
  loglevel='INFO',
):

  """Retrieve information about a classifier stored in a pickle file

  Parameters
  ----------
  classifier_path : path to pickle file with classifier dict
  loglevel : loglevel setting (default='INFO')
  """

  # remove default logger handler and add personal one
  logger.remove()
  logger.add(sys.stderr, level=loglevel)

  logger.info('Inspecting classifier pickle file')

  logger.debug(f'{locals()}')
  logger.debug(f'file location: {__file__}')

  # get directory where module is installed
  module_path = pathlib.Path(__file__).parent.parent
  logger.debug(f'module_path: {module_path}')

# -------------------------------------------------------------------------- #

  # convert folder strings to paths
  classifier_path = pathlib.Path(classifier_path).resolve()

  logger.debug(f'classifier_path: {classifier_path}')

  if not classifier_path.is_file():
    logger.error(f'Cannot find classifier_path: {classifier_path}')
    raise FileNotFoundError(f'Cannot find classifier_path: {classifier_path}')

# -------------------------------------------------------------------------- #

  # load classifier dictionary
  classifier_dict = gia.read_classifier_dict_from_pickle(
    classifier_path.as_posix()
  )

  # check that pickle file contains a dictionary
  if type(classifier_dict) is not dict:
    logger.error(
      f'Expected a classifier dictionary, but type is {type(classifier_dict)}'
    )
    raise TypeError(
      f'Expected a classifier dictionary, but type is {type(classifier_dict)}'
    )

  logger.debug('pickle file contains a classifier dictionary')
  logger.debug(f'dict keys are: {list(classifier_dict.keys())}')

  if not 'type' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `type` key')
    raise KeyError(f'classifier_dict does not contain `type` key')

  if not 'required_features' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `required_features` key')
    raise KeyError(f'classifier_dict does not contain `required_features` key')

  if not 'label_value_mapping' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `label_value_mapping` key')
    raise KeyError(
      f'classifier_dict does not contain `label_value_mapping` key'
    )

  if not 'trained_classes' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `trained_classes` key')
    raise KeyError(f'classifier_dict does not contain `trained_classes` key')

  if not 'invalid_swaths' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `invalid_swaths` key')
    raise KeyError(f'classifier_dict does not contain `invalid_swaths` key')

  if not 'info' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `info` key')
    raise KeyError(f'classifier_dict does not contain `info` key')

# -------------------------------------------------------------------------- #

  # a "gaussian_IA" type classifier must have a "gaussian_IA_params" key
  # values in this key are used to build the classifier object when needed
  # this circumvents issues with changes in the gaussin_IA_classifier module

  if classifier_dict['type'] == 'gaussian_IA':
    if not 'gaussian_IA_params' in classifier_dict.keys():
      logger.error(
        f'classifier_dict does not contain `gaussian_IA_params` key'
      )
      raise KeyError(
        f'classifier_dict does not contain `gaussian_IA_params` key'
      )
      

  # extract information for inspection output
  classifier_type     = classifier_dict['type']
  features            = classifier_dict['required_features']
  label_value_mapping = classifier_dict['label_value_mapping']
  trained_classes     = classifier_dict['trained_classes']
  invalid_swaths      = classifier_dict['invalid_swaths']
  info                = classifier_dict['info']


  print(f'\n=== CLASSIFIER ===')
  print(classifier_path)

  print('\n=== CLASSIFIER TYPE: ===')
  print(classifier_type)

  print('\n=== REQUIRED FEATURES: ===')
  for idx, feature_name in enumerate(features):
    print(f'{idx:2d} -- {feature_name}')

  print('\n=== LABEL VALUE MAPPING: ===')
  for idx, key in enumerate(label_value_mapping):
    print(f'{key} -- {label_value_mapping[key]}')

  print('\n=== TRAINED CLASSES: ===')
  print(f'{trained_classes}')

  print('\n=== INVALID SWATHS: ===')
  print(f'{invalid_swaths}')

  if 'texture_settings' in classifier_dict.keys():
    print('\n=== TEXTURE PARAMETER SETTINGS: ===')
    for idx, key in enumerate(classifier_dict['texture_settings']):
      print(f'{key}: {classifier_dict["texture_settings"][key]}')

  print('\n=== INFO: ===')
  print(f'{info}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def full_S1_image_processing_chain(
  safe_folder,
  feat_folder,
  result_folder,
  classifier_path,
  step=21,
  valid=False,
  block_size=1e6,
  overwrite=False,
  loglevel='INFO',
):

  """Fully process S1 input image: feature extraction and classification

  Parameters
  ----------
  safe_folder : path to S1 input image SAFE folder
  feat_folder : path to feature folder
  result_folder : path to result folder where labels file is placed
  classifier_path : path to pickle file with classifier dict
  step : step for GLCM calculation, if needed (default=21)
  valid : use valid mask
  block_size : number of pixels for block-wise processing (default=1e6)
  overwrite : overwrite existing files (default=False)
  loglevel : loglevel setting (default='INFO')
  """

  # remove default logger handler and add personal one
  logger.remove()
  logger.add(sys.stderr, level=loglevel)

  logger.info('Fully processing S1 input image for specified classifier')

  logger.debug(f'{locals()}')
  logger.debug(f'file location: {__file__}')

  # get directory where module is installed
  module_path = pathlib.Path(__file__).parent.parent
  logger.debug(f'module_path: {module_path}')

# -------------------------------------------------------------------------- #

  # convert folder strings to paths
  safe_folder     = pathlib.Path(safe_folder).expanduser().absolute()
  feat_folder     = pathlib.Path(feat_folder).expanduser().absolute()
  result_folder   = pathlib.Path(result_folder).expanduser().absolute()
  classifier_path = pathlib.Path(classifier_path).expanduser().absolute()

  # convert block_size and step string to integer
  block_size = int(block_size)
  step       = int(step)

  logger.debug(f'safe_folder:     {safe_folder}')
  logger.debug(f'feat_folder:     {feat_folder}')
  logger.debug(f'result_folder:   {result_folder}')
  logger.debug(f'classifier_path: {classifier_path}')
  logger.debug(f'conf.GPT:        {conf.GPT}')

  if not os.path.exists(conf.GPT):
    logger.error(f'Cannot find snap GPT executable conf.GPT: {conf.GPT}')
    raise FileNotFoundError(
      f'Cannot find snap GPT executable conf.GPT: {conf.GPT}'
    )

  if not safe_folder.is_dir():
    logger.error(f'Cannot find Sentinel-1 SAFE folder: {safe_folder}')
    raise NotADirectoryError(
      f'Cannot find Sentinel-1 SAFE folder: {safe_folder}'
    )

  if not classifier_path.is_file():
    logger.error(f'Cannot find classifier_path: {classifier_path}')
    raise FileNotFoundError(f'Cannot find classifier_path: {classifier_path}')

  # get S1 basename from safe_folder
  f_base = safe_folder.stem

  # build datestring
  date, datetime, datestring = S1.get_S1_datestring(f_base)

  # get product mode and type
  p_mode, p_type, p_pol = S1.get_S1_product_info(f_base)

  logger.debug(f'f_base:     {f_base}')
  logger.debug(f'date:       {date}')
  logger.debug(f'datetime:   {datetime}')
  logger.debug(f'datestring: {datestring}')
  logger.debug(f'p_mode:     {p_mode}')
  logger.debug(f'p_type:     {p_type}')
  logger.debug(f'p_pol:      {p_pol}')

# -------------------------------------------------------------------------- #

  # get classifier information

  logger.info('Reading classifier pickle file and extracting information')

  # load classifier dictionary
  classifier_dict = gia.read_classifier_dict_from_pickle(
    classifier_path.as_posix()
  )

  logger.debug(f'keys in classifier_dict: {classifier_dict.keys()}')

  if not 'type' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `type` key')
    raise KeyError(f'classifier_dict does not contain `type` key')

  if not 'required_features' in classifier_dict.keys():
    logger.error(f'classifier_dict does not contain `required_features` key')
    raise KeyError(f'classifier_dict does not contain `required_features` key')

  # get clf_type
  clf_type = classifier_dict['type']

  # get list of required features for classifier
  required_features = classifier_dict['required_features']

  # make list of features that need to be checked for byte order
  check_features = copy.copy(required_features)

  # get dict of texture parameters if it exists
  # should be done more robust, may currently crash if key does not exist
  if 'texture_settings' in classifier_dict.keys():

    logger.info('Reading texture settings from classifier_dict')

    key_mapping = dict()
    key_mapping['window size w']      = 'w'
    key_mapping['grey levels k']      = 'k'
    key_mapping['GLCM distance d']    = 'd'
    key_mapping['HH min for scaling'] = 'HH_min'
    key_mapping['HH max for scaling'] = 'HH_max'
    key_mapping['HV min for scaling'] = 'HV_min'
    key_mapping['HV max for scaling'] = 'HV_max'

    texture_params_dict = dict()

    for key in key_mapping:
      if key in classifier_dict['texture_settings'].keys():
        texture_params_dict[key_mapping[key]] = \
          classifier_dict['texture_settings'][key]
      else:
        logger.warning(f'texture_settings dict does not contain `{key}` key')

      logger.debug(f'{texture_params_dict}')

# -------------------------------------------------------------------------- #

  # always extract the following features

  logger.info('Extracting standard features: incident angle and swath mask')

  # incident angle
  S1_feat.get_S1_IA(
    safe_folder.as_posix(),
    feat_folder.as_posix(),
    overwrite=overwrite,
    loglevel=loglevel
  )

  # swath mask
  S1_feat.get_S1_swath_mask(
    safe_folder.as_posix(),
    feat_folder.as_posix(),
    overwrite=overwrite,
    loglevel=loglevel
  )

  check_features.extend(['IA','swath_mask'])

# -------------------------------------------------------------------------- #

  # extract texture features needed for classifier

  # check if and what kind of GLCM texture is needed
  GLCM_texture = False
  db           = False
  db_string    = ''
  GLCM_from_HH = False
  GLCM_from_HV = False

  for feat in required_features:
    if 'GLCM' in feat:
      GLCM_texture = True
      if 'db' in feat:
        db = True
        db_string = '_db'
      if 'HH' in feat:
        GLCM_from_HH = True
      if 'HV' in feat:
        GLCM_from_HV = True

  # get GLCM texture if needed
  if GLCM_texture:

    logger.info('GLCM texture is required')
    logger.info(f'texture_params_dict: {texture_params_dict}')


    # get GLCM features from HH
    if GLCM_from_HH:
   
      logger.info(f'Extracting Sigma0_HH{db_string} for GLCM calculation')

      S1_feat.get_S1_HH(
        safe_folder.as_posix(),
        feat_folder.as_posix(),
        ML='1x1',
        dB=db,
        overwrite=True,
        loglevel=loglevel
      )

      logger.info(f'Extracting GLCM texture from Sigma0_HH{db_string}')

      gen_feat.get_texture_GLCM(
        feat_folder / f'Sigma0_HH{db_string}.img',
        feat_folder.as_posix(),
        w = texture_params_dict['w'],
        k = texture_params_dict['k'],
        d = texture_params_dict['d'],
        step = step,
        img_vmin = texture_params_dict['HH_min'],
        img_vmax = texture_params_dict['HH_max'],
        overwrite=True,
        loglevel=loglevel
      )

      logger.info(f'Removing Sigma0_HH{db_string}')
      os.remove(feat_folder / f'Sigma0_HH{db_string}.img')
      os.remove(feat_folder / f'Sigma0_HH{db_string}.hdr')

    # get GLCM features from HV
    if GLCM_from_HV:

      logger.info(f'Extracting Sigma0_HV{db_string} for GLCM calculation')

      S1_feat.get_S1_HV(
        safe_folder.as_posix(),
        feat_folder.as_posix(),
        ML='1x1',
        dB=db,
        overwrite=True,
        loglevel=loglevel
      )

      logger.info(f'Extracting GLCM texture from Sigma0_HV{db_string}')

      gen_feat.get_texture_GLCM(
        feat_folder / f'Sigma0_HV{db_string}.img',
        feat_folder.as_posix(),
        w = texture_params_dict['w'],
        k = texture_params_dict['k'],
        d = texture_params_dict['d'],
        step = step,
        img_vmin = texture_params_dict['HV_min'],
        img_vmax = texture_params_dict['HV_max'],
        overwrite=True,
        loglevel=loglevel
      )

      logger.info(f'Removing Sigma0_HV{db_string}')
      os.remove(feat_folder / f'Sigma0_HV{db_string}.img')
      os.remove(feat_folder / f'Sigma0_HV{db_string}.hdr')


  # GLCM features are on default saved with '_step_{step}' in file name
  # rename the required features
  for feat in required_features:

    logger.info('Renaming GLCM features to required naming convention')

    if 'GLCM' in feat:
      os.rename(
        feat_folder / f'{feat}_step_{step}.img',
        feat_folder / f'{feat}.img'
      )
      os.rename(
        feat_folder / f'{feat}_step_{step}.hdr',
        feat_folder / f'{feat}.hdr'
      )

# -------------------------------------------------------------------------- #

  # extract intensity features needed for classifier

  if 'Sigma0_HH_db' in required_features:

    logger.info('Extracting Sigma0_HH_db')

    S1_feat.get_S1_HH(
      safe_folder.as_posix(),
      feat_folder.as_posix(),
      ML='1x1',
      dB=True,
      overwrite=True,
      loglevel=loglevel
    )

  if 'Sigma0_HV_db' in required_features:

    logger.info('Extracting Sigma0_HV_db')

    S1_feat.get_S1_HV(
      safe_folder.as_posix(),
      feat_folder.as_posix(),
      ML='1x1',
      dB=True,
      overwrite=True,
      loglevel=loglevel
    )

# -------------------------------------------------------------------------- #

  # create a valid mask, if required

  if valid:

    logger.info('Creating valid mask')

    check_features.append('valid')

    if not 'invalid_swaths' in classifier_dict.keys():
      logger.warning(f'classifier_dict does not contain `invalid_swaths` key')
      logger.warning(f'setting swath 0 as invalid')
      invalid_swaths = '0'
    else:
      invalid_swaths = classifier_dict['invalid_swaths']

    logger.debug(f'invalid_swaths: {invalid_swaths}')

    S1_feat.make_S1_valid_mask(
      feat_folder / 'swath_mask.img',
      feat_folder,
      invalid_swaths = invalid_swaths,
      overwrite=True
    )

    # sub-step valid mask if needed
    if GLCM_texture:
      valid_mask = rwi.get_all_bands((feat_folder / 'valid.img').as_posix())
      valid_mask_sub = valid_mask[::step, ::step]
      rwi.write_img(
        (feat_folder / 'valid.img').as_posix(),
        valid_mask_sub,
        as_type=gdal.GDT_Byte,
        overwrite=True,
      )

# -------------------------------------------------------------------------- #

  # check byte order of all features for memory mapping

  for check_feature in check_features:
    logger.info(f'Checking byte order of feature: {check_feature}')

    rwi.check_envi_byte_order(
      (feat_folder / f'{check_feature}.img').as_posix()
    )

# -------------------------------------------------------------------------- #

  # perform classification
  probs_out, Mahal_out, shape = classify_image(
    feat_folder,
    result_folder,
    classifier_path,
    valid = valid,
    block_size = block_size,
    overwrite = overwrite,
    loglevel = loglevel
  )
  
  return probs_out, Mahal_out, shape

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classify_image.py> ----
