# ---- This is <classification_utils.py> ----

"""
Helper functions for classification module
"""

from sys import byteorder
from pathlib import Path

from loguru import logger

from osgeo.gdal import Open, GA_ReadOnly

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def check_image_byte_order(img_path):
    """Check byte order of the input image and return whether or not it matches the system byte order
       
    Parameters
    ----------
    img_path : path to input image

    Returns
    -------
    byteswap_needed : Boolean defining whether byteswap is needed for input image on current system
    """ 

    # make sure that input image path is pathlib object
    img_path = Path(img_path)

    # check that img_path exists
    if not img_path.is_file():
        logger.error(f'Cannot find img_path: {img_path}')
        raise FileNotFoundError(f'Cannot find img_path: {img_path}')

# --------------------- #

    logger.debug(f'Checking byte order for image file: {img_path}')

    if img_path.suffix == '.img':
        logger.debug("img_path is of type 'img'")

        # build path to hdr file
        hdr_path = img_path.parent / f'{img_path.stem}.hdr'

        with open(hdr_path.as_posix()) as ff:
            header_contents = ff.read().splitlines()
        for header_line in header_contents:
            if 'byte order' in header_line:
                logger.debug(f'header line with byte order is: {header_line}')
                img_byte_order = int(header_line[-1])

    else:
        logger.error(f'Current image type is not implemented: {img_path.suffix}')
        raise TypeError(f'Current image type is not implemented: {img_path.suffix}')

    logger.debug(f'img_byte_order: {img_byte_order}')

# --------------------- #

    # get system byte order
    system_byte_order = byteorder
    logger.debug(f'system_byte_order: {system_byte_order}')

    # convert system_byte_order to integer value
    if system_byte_order == 'little':
        system_byte_order = 0
    elif system_byte_order == 'big':
        system_byte_order = 1
    else:
        logger.error(f'Unknown system byte order: {system_byte_order}')
        raise ValueError(f'Unknown system byte order: {system_byte_order}')

    logger.debug(f'converted system_byte_order to: {system_byte_order}')

# --------------------- #

    # compare system_byte_order and img_byte_order
    if img_byte_order == system_byte_order:
        logger.debug('img_byte_order and system_byte_order match')
        byteswap_needed = False
    elif img_byte_order != system_byte_order:
        logger.debug('img_byte_order and system_byte_order do not match')
        byteswap_needed = True

    logger.debug(f'byteswap_needed for memory mapping set to: {byteswap_needed}')

# --------------------- #

    return byteswap_needed

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def get_image_dimensions(img_path):
    """Get dimensions of input image
       
    Parameters
    ----------
    img_path : path to input image

    Returns
    -------
    Nx : image raster X size
    Ny : image raster Y size
    """ 


    # make sure that input image path is pathlib object
    img_path = Path(img_path)

    # check that img_path exists
    if not img_path.is_file():
        logger.error(f'Cannot find img_path: {img_path}')
        raise FileNotFoundError(f'Cannot find img_path: {img_path}')

# --------------------- #

    # get Nx and Ny from first required feature
    ds = Open(img_path.as_posix(), GA_ReadOnly)
    Nx = ds.RasterXSize
    Ny = ds.RasterYSize
    ds = []

    logger.debug(f'Read image dimensions: {Nx,Ny}')

# --------------------- #

    return Nx, Ny

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classification_utils.py> ----
