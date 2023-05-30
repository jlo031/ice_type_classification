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

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def classify_S1_image_from_feature_folder(
    feat_folder,
    result_folder,
    classifier_model_path,
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

    logger.debug(f'feat_folder:           {feat_folder}')
    logger.debug(f'result_folder:         {result_folder}')
    logger.debug(f'classifier_model_path: {classifier_model_path}')

    if not feat_folder.is_dir():
        logger.error(f'Cannot find feat_folder: {feat_folder}')
        raise NotADirectoryError(f'Cannot find feat_folder: {feat_folder}')

    if not classifier_model_path.is_file():
        logger.error(f'Cannot find classifier_model_path: {classifier_model_path}')
        raise FileNotFoundError(f'Cannot find classifier_model_path: {classifier_model_path}')

    # get input basename from feat_folder
    f_base = feat_folder.stem

    # define output file names and paths
    result_basename = f_base + '_labels'
    result_path     = result_folder / f'{result_basename}.img'
    result_path_hdr = result_folder / f'{result_basename}.hdr'

    # for Kristian (fix later)
    ##result_path_Mahal     = result_folder / f'{result_basename}_Mahal.img'
    ##result_path_hdr_Mahal = result_folder / f'{result_basename}_Mahal.hdr'
    ##result_path_probs     = result_folder / f'{result_basename}_probs.img'
    ##result_path_hdr_probs = result_folder / f'{result_basename}_probs.hdr'

    logger.debug(f'result_path: {result_path}')

    # check if outfile already exists
    if result_path.is_file() and not overwrite:
        logger.info('Output files already exist, use `-overwrite` to force')
        return
    elif result_path.is_file() and overwrite:
        logger.info('Removing existing output file and classifying again')
        result_path.unlink()
        result_path_hdr.unlink()

# -------------------------------------------------------------------------- #

    # get system byte order for memory mapping

    system_byte_order = sys.byteorder

    # convert system_byte_order to integer
    if system_byte_order == 'little':
        system_byte_order = 0
    elif system_byte_order == 'big':
        system_byte_order = 1

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

    logger.debug(f'clf_type:          {clf_type}')
    logger.debug(f'required_features: {required_features}')

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
    ds = gdal.Open((feat_folder / f'{required_features[0]}.img').as_posix(), gdal.GA_ReadOnly)
    Nx = ds.RasterXSize
    Ny = ds.RasterYSize
    ds = []


    # check that all required features have same dimensions
    for f in required_features:
        ds =  gdal.Open((feat_folder / f'{f}.img').as_posix(), gdal.GA_ReadOnly)
        Nx_current = ds.RasterXSize
        Ny_current = ds.RasterYSize
        ds = []
        if not Nx == Nx_current or not Ny == Ny_current:
            logger.error(f'Image dimensions of required features do not match')
            raise ValueError(f'Image dimensions of required features do not match')

# -------------------------------------------------------------------------- #

    # CHECK VALID AND IA MASK IF REQUIRED

    if valid_mask:
        logger.info('Using valid mask')

        # check that valid.img exists
        if 'valid.img' not in existing_features:
            logger.error('Cannot find valid mask: valid.img')
            logger.error('Unset `-valid_mask` flag, if you do not want to use a valid mask')
            raise FileNotFoundError('Cannot find valid mask: valid.img')

        else:
            # get valid_mask dimensions
            ds =  gdal.Open((feat_folder / 'valid.img').as_posix(), gdal.GA_ReadOnly)
            Nx_valid = ds.RasterXSize
            Ny_valid = ds.RasterYSize
            ds = []
            # check that valid_mask dimensions match feature dimensions
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




    if clf_type == 'gaussian_IA':
        logger.info('Classifier is using IA information')

        # check that IA.img exists
        if 'IA.img' not in existing_features:
            logger.error(f'Cannot find IA image: IA.img')
            raise FileNotFoundError(f'Cannot find IA image: IA.img')

        else:
            # get IA dimensions
            ds =  gdal.Open((feat_folder / 'IA.img').as_posix(), gdal.GA_ReadOnly)
            Nx_IA = ds.RasterXSize
            Ny_IA = ds.RasterYSize
            ds = []
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
        clf = gia.make_gaussian_IA_clf_object_from_params_dict(classifier_dict['gaussian_IA_params'])

    elif clf_type =='gaussian':
        logger.debug(f'clf_valid_mask_data_typetype: {clf_type}')
        logger.debug('classifier_dict only contains clf parameters')
        logger.debug('Setting up classifier object from these parameters')
        clf = gia.make_gaussian_clf_object_from_params_dict(classifier_dict['gaussian_params'])

    else:
        ##clf = classifier_dict['clf_object']
        logger.error('This clf type is not implemented yet')
        raise NotImplementedError('This clf type is not implemented yet')

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
        ).byteswap()
    else:
        IA_mask = np.zeros(N).astype(int)

    # memory map valid mask if required, set valid mask to 1 otherwise
    if valid_mask:
        logger.debug('Memory mapping valid_mask')
        valid_mask = np.memmap(
            (feat_folder / 'valid.img').as_posix(), 
            dtype=valid_mask_data_type, mode='r', shape=(N)
        ).byteswap()
    else:
        logger.debug('Setting valid_mask to 1')
        valid_mask = np.ones(N).astype(int)





    for f in required_features:

        logger.debug(f'Checking byte order for current feature: {f}')

        # get current feature byte order
        hdr_file = feat_folder/f'{f}.hdr'
        with open(hdr_file.as_posix()) as ff:
            header_contents = ff.read().splitlines()
        for header_line in header_contents:
            if 'byte order' in header_line:
                logger.debug(header_line)
                img_byte_order = int(header_line[-1])


        # check if img and system byte orders match
        if img_byte_order == system_byte_order:
            logger.debug('Image byte order matches system byte order for current feature')
            data_dict[f] = np.memmap(
                f'{feat_folder.as_posix()}/{f}.img', 
                dtype=np.float32, mode='r', shape=(N)
            )
        elif img_byte_order != system_byte_order:
            logger.debug('Image byte order does not match system byte order for current feature')
            data_dict[f] = np.memmap(
                f'{feat_folder.as_posix()}/{f}.img', 
                dtype=np.float32, mode='r', shape=(N)
            ).byteswap()


        """
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
            ).byteswap()
        """

# -------------------------------------------------------------------------- #

    # initialize labels and probabilities
    labels_img = np.zeros(N)
    ##Mahal_img  = np.zeros(N)
    ##probs_img  = np.zeros(N)

    # find number of blocks from block_size
    n_blocks   = int(np.ceil(N/block_size))

    # logger
    logger.info('Performing block-wise processing of memory-mapped data')
    logger.info(f'block-size: {block_size}')
    logger.info(f'n_blocks:   {n_blocks}')

    # for progress report at every 10%
    log_percs   = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    perc_blocks = np.ceil(np.array(n_blocks)*log_percs).astype(int)

# -------------------------------------------------------------------------- #

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


        # calculate Mahalanobis distance for each class
        ##Mahal_img[idx_start:idx_end][valid_block==1] = Mahalanobis_distance(X_block_valid, classifier_dict)


        # predict labels where valid==1
        if clf_type == 'gaussian_IA':
            labels_img[idx_start:idx_end][valid_block==1], dummy = \
                clf.predict(X_block_valid, IA_block_valid)

        elif clf_type == 'gaussian':
            labels_img[idx_start:idx_end][valid_block==1], dummy = \
                clf.predict(X_block_valid)

        else:
            logger.error('This clf type is not implemented yet')

        # set labels to 0 where valid==0
        labels_img[idx_start:idx_end][valid_block==0] = 0

    logger.info('Finished classification')

    # reshape to image geometry
    labels_img = np.reshape(labels_img,shape)
    ##Mahal_img  = np.reshape(Mahal_img,shape)
    ##probs_img  = np.reshape(probs_img,shape)

# -------------------------------------------------------------------------- #

    # create result_folder if needed
    result_folder.mkdir(parents=True, exist_ok=True)

    # get drivers
    output = gdal.GetDriverByName('Envi').Create(result_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)

    ##output_Mahal = gdal.GetDriverByName('Envi').Create(result_path_Mahal.as_posix(), Nx, Ny, n_classes, gdal.GDT_Float32)

    ##output_probs = gdal.GetDriverByName('Envi').Create(result_path_probs.as_posix(), Nx, Ny, n_classes, gdal.GDT_Float32)


    # write labels_img to band 1
    output.GetRasterBand(1).WriteArray(labels_img)

    # flush
    output.FlushCache()

    # logger
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
