# ---- This is <classification.py> ----

"""
Module for forward classification of satellite images
""" 

import argparse
import os
import sys
import pathlib
import shutil
import copy

from loguru import logger

import numpy as np

from osgeo import gdal

import ice_type_classification.gaussian_IA_classifier as gia

import ice_type_classification.uncertainty_utils as uncertainty_utils
import ice_type_classification.classification_utils as classification_utils

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def classify_S1_image_from_feature_folder(
    feat_folder,
    result_folder,
    classifier_model_path,
    uncertainties = True,
    uncertainty_dict = [],
    valid_mask = False,
    block_size = 1e6,
    overwrite = False,
    loglevel = 'INFO',
):

    """Classify S1 input image

    Parameters
    ----------
    feat_folder : path to input feature folder
    result_folder : path to result folder where labels file is placed
    classifier_model_path : path to pickle file with classifier model dict
    uncertainties : estimate apost and mahal uncertainties (default True)
    uncertainty_dict : dictionary with parameters for uncertainty estimation
    valid_mask : use valid mask
    block_size : number of pixels for block-wise processing (default=1e6)
    overwrite : overwrite existing files (default=False)
    loglevel : loglevel setting (default='INFO')
    """

    # remove default logger handler and add personal one
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    logger.info('Classifying input image')

# -------------------------------------------------------------------------- #

    # convert folder strings to paths
    feat_folder           = pathlib.Path(feat_folder).expanduser().absolute()
    result_folder         = pathlib.Path(result_folder).expanduser().absolute()
    classifier_model_path = pathlib.Path(classifier_model_path).expanduser().absolute()

    # convert block_size string to integer
    block_size = int(block_size)

    logger.debug(f'feat_folder: {feat_folder}')
    logger.debug(f'result_folder: {result_folder}')
    logger.debug(f'classifier_model_path: {classifier_model_path}')

    if not feat_folder.is_dir():
        logger.error(f'Cannot find feat_folder: {feat_folder}')
        raise NotADirectoryError(f'Cannot find feat_folder: {feat_folder}')

    if not classifier_model_path.is_file():
        logger.error(f'Cannot find classifier_model_path: {classifier_model_path}')
        raise FileNotFoundError(f'Cannot find classifier_model_path: {classifier_model_path}')

    # get input basename from feat_folder
    f_base = feat_folder.stem

    logger.debug(f'f_base: {f_base}')

    # define output file names and paths
    result_basename = f_base + '_labels'
    result_path     = result_folder / f'{result_basename}.img'
    result_path_hdr = result_folder / f'{result_basename}.hdr'
    result_path_mahal     = result_folder / f'{f_base}_mahal_uncertainty.img'
    result_path_mahal_hdr = result_folder / f'{f_base}_mahal_uncertainty.hdr'
    result_path_apost     = result_folder / f'{f_base}_apost_uncertainty.img'
    result_path_apost_hdr = result_folder / f'{f_base}_apost_uncertainty.hdr'

    logger.debug(f'result_path:       {result_path}')
    logger.debug(f'result_path_mahal: {result_path_mahal}')
    logger.debug(f'result_path_apost: {result_path_apost}')

    # check if main outfile already exists
    if result_path.is_file() and not overwrite:
        logger.info('Output files already exist, use `-overwrite` to force')
        return
    elif result_path.is_file() and overwrite:
        logger.info('Removing existing output file and classifying again')
        result_path.unlink()
        result_path_hdr.unlink()
        result_path_mahal.unlink(missing_ok=True)
        result_path_mahal_hdr.unlink(missing_ok=True)
        result_path_apost.unlink(missing_ok=True)
        result_path_apost_hdr.unlink(missing_ok=True)

# -------------------------------------------------------------------------- #

    # GET BASIC CLASSIFIER INFO

    # load classifier dictionary
    classifier_dict = gia.read_classifier_dict_from_pickle(classifier_model_path.as_posix())

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

    logger.info(f'clf_type: {clf_type}')
    logger.info(f'required_features: {required_features}')

# ---------------------------------- #

    # BUILD CLASSIFIER OBJECT ACCORDING TO CLASSIFIER DICT

    # check classifier type
    # for gaussian or gaussian_IA:
    #     - build the actual clf object from the saved parameters
    # for other types:
    #      - set the clf object directly from classifier_dict
    #      - import the necessary modules from sklearn

    if clf_type == 'gaussian_IA':
        logger.debug(f'clf_type: {clf_type}')
        logger.debug('classifier_dict should only contain clf parameters')
        logger.debug('building classifier object from these parameters')
        clf = gia.make_gaussian_IA_clf_object_from_params_dict(classifier_dict['gaussian_IA_params'])

    elif clf_type =='gaussian':
        logger.debug(f'clf_valid_mask_data_typetype: {clf_type}')
        logger.debug('classifier_dict should only contain clf parameters')
        logger.debug('building classifier object from these parameters')
        clf = gia.make_gaussian_clf_object_from_params_dict(classifier_dict['gaussian_params'])

    else:
        logger.error('This clf type is not implemented yet')
        raise NotImplementedError('This clf type is not implemented yet')

# -------------------------------------------------------------------------- #

    # PREPARE UNCERTAINTY ESTIMATION

    if uncertainties:
        logger.info('Uncertainties is set to "True"')

        # extract clf parameters needed for uncertainties
        if clf_type == 'gaussian_IA':
            mu_vec_all_classes  = classifier_dict['gaussian_IA_params']['mu']
            cov_mat_all_classes = classifier_dict['gaussian_IA_params']['Sigma']
            n_classes           = int(classifier_dict['gaussian_IA_params']['n_class'])
            n_features          = int(classifier_dict['gaussian_IA_params']['n_feat'])
            IA_0                = classifier_dict['gaussian_IA_params']['IA_0']
            IA_slope            = classifier_dict['gaussian_IA_params']['b']
        elif clf_type =='gaussian':
            mu_vec_all_classes   = classifier_dict['gaussian_params']['mu']
            cov_vmat_all_classes = classifier_dict['gaussian_params']['Sigma']
            n_classes            = int(classifier_dict['gaussian_params']['n_class'])
            n_features           = int(classifier_dict['gaussian_params']['n_feat'])
        else:
            logger.error('This clf type is not implemented yet')
            raise NotImplementedError('This clf type is not implemented yet')


        # set default values for uncertainty estimation
        uncertainty_params = dict()
        uncertainty_params['apost_uncertainty_measure'] = 'Entropy'
        uncertainty_params['DO_apost_uncertainty'] = True
        uncertainty_params['DO_mahal_uncertainty'] = True
        uncertainty_params['discrete_uncertainty'] = True
        uncertainty_params['mahal_thresh_min'] = 6
        uncertainty_params['mahal_thresh_max'] = 12
        uncertainty_params['mahal_discrete_thresholds'] = np.array([6, 8, 10, 12])
        uncertainty_params['apost_discrete_thresholds'] = ['default']

        valid_uncertainty_keys = uncertainty_params.keys()


        # user uncertainty_dict if given
        if uncertainty_dict == []:
            logger.info('Using default parameters for uncertainty estimation')

        elif type(uncertainty_dict) == dict:
            logger.info('Using parameters from uncertainty_dict for uncertainty estimation')

            uncertainty_keys = uncertainty_dict.keys()
            logger.debug(f'uncertainty_dict.keys(): {uncertainty_keys}')

            # overwrite uncertainty_params with correct values from uncertainty_dict
            for key in uncertainty_keys:
                if key in valid_uncertainty_keys:
                    logger.debug(f'overwriting default value for "{key}" with: {uncertainty_dict[key]}')
                    uncertainty_params[key] = uncertainty_dict[key]
                else:
                    logger.warning(f'uncertainty_dict key "{key}" is unknown and will not be used')

        else:
            logger.warning(f'Expected "uncertainty_dict" of type "dict", but found type "{type(uncertainty_dict)}"')
            logger.warning('Using default parameters for uncertainty estimation')

# -------------------------------------------------------------------------- #

    # CHECK EXISTING AND REQUIRED FEATURES

    # get list of existing features in feat_folder
    existing_features = sorted([f for f in os.listdir(feat_folder) if f.endswith('img')])

    # check that all required_features exist
    for f in required_features:
        if f'{f}.img' not in existing_features:
            logger.error(f'Cannot find required feature: {f}')
            raise FileNotFoundError(f'Cannot find required feature: {f}')

    # get Nx and Ny from first required feature
    Nx, Ny = classification_utils.get_image_dimensions((feat_folder / f'{required_features[0]}.img').as_posix())
    shape  = (Ny, Nx)
    N      = Nx*Ny
    logger.info(f'Image dimensions: Nx={Nx}, Ny={Ny}')
    logger.info(f'Image shape: {shape}')
    logger.info(f'Total number of pixels: {N}')

    # check that all required features have same dimensions
    for f in required_features:
        Nx_current, Ny_current = classification_utils.get_image_dimensions(  (feat_folder / f'{f}.img').as_posix() )
        if not Nx == Nx_current or not Ny == Ny_current:
            logger.error(f'Image dimensions of required features do not match')
            raise ValueError(f'Image dimensions of required features do not match')

# -------------------------------------------------------------------------- #

    # CHECK VALID MASK

    if valid_mask:
        logger.info('Using valid mask')

        # check that valid.img exists
        if 'valid.img' not in existing_features:
            logger.error('Cannot find valid mask: valid.img')
            logger.error('Unset `-valid_mask` flag, if you do not want to use a valid mask')
            raise FileNotFoundError('Cannot find valid mask: valid.img')

        else:
            # check that valid_mask dimensions match feature dimensions
            Nx_valid, Ny_valid = classification_utils.get_image_dimensions((feat_folder / 'valid.img').as_posix())
            if not Nx == Nx_valid or not Ny == Ny_valid:
                logger.error(f'valid_mask dimensions do not match featured imensions')
                raise ValueError(f'valid_mask dimensions do not match featured imensions')

            # get valid_mask data type
            valid_data_type_in = gdal.Open((feat_folder / 'valid.img').as_posix(), gdal.GA_ReadOnly).GetRasterBand(1).DataType
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
        logger.info('Not using valid mask, set `-valid_mask` if wanted')

# ---------------------------------- #

    # CHECK IA MASK

    if clf_type == 'gaussian_IA':
        logger.info('Classifier requires IA information')

        # check that IA.img exists
        if 'IA.img' not in existing_features:
            logger.error(f'Cannot find IA image: IA.img')
            raise FileNotFoundError(f'Cannot find IA image: IA.img')

        else:
            # check that IA dimensions match feature dimensions
            Nx_IA, Ny_IA = classification_utils.get_image_dimensions((feat_folder / 'IA.img').as_posix())
            if not Nx == Nx_IA or not Ny == Ny_IA:
                logger.error(f'IA dimensions do not match featured imensions')
                raise ValueError(f'IA dimensions do not match featured imensions')

    else:
        logger.info('Classifier is not using IA information')

# -------------------------------------------------------------------------- #

    # initialize data dict and/or data list
    data_dict = dict()
    data_list = []

    # logger
    logger.info('Memory mapping all required data')

# --------------------- #

    # memory map IA if required, set empty otherwise

    if clf_type=='gaussian_IA':

        logger.debug('Memory mapping IA')

        # check of byteswap is needed
        byteswap_needed = classification_utils.check_image_byte_order(feat_folder/f'IA.img')

        if byteswap_needed:
            IA_mask = np.memmap((feat_folder / 'IA.img').as_posix(), dtype=np.float32, mode='r', shape=(N)).byteswap()
        elif not byteswap_needed:
            IA_mask = np.memmap((feat_folder / 'IA.img').as_posix(), dtype=np.float32, mode='r', shape=(N))

    else:
        IA_mask = np.zeros(N).astype(int)

# --------------------- #

    # memory map valid mask if required, set empty otherwise

    if valid_mask:

        logger.debug('Memory mapping valid mask')

        # check of byteswap is needed
        byteswap_needed = classification_utils.check_image_byte_order(feat_folder/f'valid.img')

        if byteswap_needed:
            valid_mask = np.memmap((feat_folder / 'valid.img').as_posix(), dtype=valid_mask_data_type, mode='r', shape=(N)).byteswap()
        elif not byteswap_needed:
            valid_mask = np.memmap((feat_folder / 'valid.img').as_posix(), dtype=valid_mask_data_type, mode='r', shape=(N))

    else:
        logger.debug('Setting valid_mask to 1')
        valid_mask = np.ones(N).astype(int)

# --------------------- #

    for f in required_features:

        logger.debug(f'Memory mapping feature: {f}')

        # check of byteswap is needed
        byteswap_needed = classification_utils.check_image_byte_order(feat_folder/f'{f}.img')

        if byteswap_needed:
            data_dict[f] = np.memmap(f'{feat_folder.as_posix()}/{f}.img', dtype=np.float32, mode='r', shape=(N)).byteswap()
        elif not byteswap_needed:
            data_dict[f] = np.memmap(f'{feat_folder.as_posix()}/{f}.img', dtype=np.float32, mode='r', shape=(N))
 
# -------------------------------------------------------------------------- #

    # initialize labels and probabilities
    labels_img = np.zeros(N)

    if uncertainties:
        # for uncertainties
        mahal_img  = np.zeros((N,n_classes))
        mahal_img.fill(np.nan)
        probs_img  = np.zeros((N,n_classes))
        probs_img.fill(np.nan)

    # find number of blocks from block_size
    n_blocks   = int(np.ceil(N/block_size))

    # logger
    logger.info('Performing block-wise processing of memory-mapped data')
    logger.info(f'block-size: {block_size}')
    logger.info(f'Number of blocks: {n_blocks}')

    # for progress report at every 10%
    log_percs   = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    perc_blocks = np.ceil(np.array(n_blocks)*log_percs).astype(int)

# -------------------------------------------------------------------------- #

    # CLASSIFY

    # loop over all blocks
    for block in np.arange(n_blocks):

        # logger
        logger.debug(f'Processing block {block+1} of {n_blocks}')

        if block in perc_blocks:
            logger.info(f'..... {int(log_percs[np.where(perc_blocks==block)[0][0]]*100)}%')

        # get idx of current block
        idx_start = block*block_size
        idx_end   = (block+1)*block_size

        # select IA and valid mask for current block
        IA_block    = IA_mask[idx_start:idx_end]
        valid_block = valid_mask[idx_start:idx_end]

        # select all features for current block (make sure order is correct)
        X_block_list = []    
        for key in required_features:
            X_block_list.append(data_dict[key][idx_start:idx_end])

        # stack all selected features to array (N,n_feat)
        X_block= np.stack(X_block_list,1)   

        # select only valid part of block
        X_block_valid  = X_block[valid_block==1]
        IA_block_valid = IA_block[valid_block==1]


        # predict labels where valid==1
        if clf_type == 'gaussian_IA':


            
            if uncertainties:
                labels_img[idx_start:idx_end][valid_block==1], probs_img[idx_start:idx_end][valid_block==1] = clf.predict(X_block_valid, IA_block_valid)

                # for uncertainties
                logger.debug('Estimating mahal_img for current block')
                mahal_img[idx_start:idx_end][valid_block==1] = uncertainty_utils.get_mahalanobis_distance(X_block_valid, mu_vec_all_classes, cov_mat_all_classes, IA_test=IA_block_valid, IA_0=IA_0, IA_slope=IA_slope)
	
            else:
                labels_img[idx_start:idx_end][valid_block==1], _ = clf.predict(X_block_valid, IA_block_valid)



        elif clf_type == 'gaussian':

            if uncertainties:
                labels_img[idx_start:idx_end][valid_block==1],probs_img[idx_start:idx_end][valid_block==1] = clf.predict(X_block_valid)

                # for uncertainties
                logger.debug('Estimating mahal_img for current block')
                mahal_img[idx_start:idx_end][valid_block==1] = uncertainty_utils.get_mahalanobis_distance(X_block_valid, mu_vec_all_classes, cov_mat_all_classes)

            else:
                labels_img[idx_start:idx_end][valid_block==1], _ = clf.predict(X_block_valid)



        else:
            logger.error('This clf type is not implemented yet')

        # set labels to 0 where valid==0
        labels_img[idx_start:idx_end][valid_block==0] = 0

    logger.info('Finished classification')

# -------------------------------------------------------------------------- #

    if uncertainties:

        logger.info('Estimating apost and mahal uncertainties')

        uncertainty_apost, uncertainty_mahal = uncertainty_utils.uncertainty(
            probs_img,
            mahal_img,
            n_features,
            apost_uncertainty_meausure = uncertainty_params['apost_uncertainty_measure'],
            DO_apost_uncertainty = uncertainty_params['DO_apost_uncertainty'],
            DO_mahalanobis_uncertainty = uncertainty_params['DO_mahal_uncertainty'],
            discrete_uncertainty = uncertainty_params['discrete_uncertainty'],
            mahal_thresh_min = uncertainty_params['mahal_thresh_min'],
            mahal_thresh_max = uncertainty_params['mahal_thresh_max'],
            mahal_discrete_thresholds = uncertainty_params['mahal_discrete_thresholds'],
            apost_discrete_thresholds = uncertainty_params['apost_discrete_thresholds']
        )

# -------------------------------------------------------------------------- #

    # reshape to image geometry
    labels_img = np.reshape(labels_img, shape)

    if uncertainties:
        if uncertainty_mahal is not False:
            uncertainty_mahal  = np.reshape(uncertainty_mahal, shape)
        if uncertainty_apost is not False:
            uncertainty_apost  = np.reshape(uncertainty_apost, shape)

        logger.info('Finished uncertainty estimation')

# -------------------------------------------------------------------------- #

    # create result_folder if needed
    result_folder.mkdir(parents=True, exist_ok=True)

    # write labels
    output_labels = gdal.GetDriverByName('Envi').Create(result_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)
    output_labels.GetRasterBand(1).WriteArray(labels_img)
    output_labels.FlushCache()


    # write uncertainties
    if uncertainties:
        if uncertainty_mahal is not False:
            output_mahal = gdal.GetDriverByName('Envi').Create(result_path_mahal.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
            output_mahal.GetRasterBand(1).WriteArray(uncertainty_mahal)
            output_mahal.FlushCache()
        if uncertainty_apost is not False:
            output_apost = gdal.GetDriverByName('Envi').Create(result_path_apost.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
            output_apost.GetRasterBand(1).WriteArray(uncertainty_apost)
            output_apost.FlushCache()

    logger.info(f'Result writtten to {result_path}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def inspect_classifier_pickle(
    classifier_model_path,
    loglevel='INFO',
):

    """Retrieve information about a classifier stored in a pickle file

    Parameters
    ----------
    classifier_mode_path : path to pickle file with classifier dict
    loglevel : loglevel setting (default='INFO')
    """

    # remove default logger handler and add personal one
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    logger.info('Inspecting classifier pickle file')

# -------------------------------------------------------------------------- #

    # convert folder strings to paths
    classifier_model_path = pathlib.Path(classifier_model_path).resolve()

    logger.debug(f'classifier_model_path: {classifier_model_path}')

    if not classifier_model_path.is_file():
        logger.error(f'Cannot find classifier_model_path: {classifier_model_path}')
        raise FileNotFoundError(f'Cannot find classifier_model_path: {classifier_model_path}')

# -------------------------------------------------------------------------- #

    # load classifier dictionary
    classifier_dict = gia.read_classifier_dict_from_pickle(classifier_model_path.as_posix())

    # check that pickle file contains a dictionary
    if type(classifier_dict) is not dict:
        logger.error(f'Expected a classifier dictionary, but type is {type(classifier_dict)}')
        raise TypeError(f'Expected a classifier dictionary, but type is {type(classifier_dict)}')

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
        raise KeyError(f'classifier_dict does not contain `label_value_mapping` key')

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
            logger.error(f'classifier_dict does not contain `gaussian_IA_params` key')
            raise KeyError(f'classifier_dict does not contain `gaussian_IA_params` key')
      

    # extract information for inspection output
    classifier_type     = classifier_dict['type']
    features            = classifier_dict['required_features']
    label_value_mapping = classifier_dict['label_value_mapping']
    trained_classes     = classifier_dict['trained_classes']
    invalid_swaths      = classifier_dict['invalid_swaths']
    info                = classifier_dict['info']


    print(f'\n=== CLASSIFIER ===')
    print(classifier_model_path)

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

# ---- End of <classifcation.py> ----
