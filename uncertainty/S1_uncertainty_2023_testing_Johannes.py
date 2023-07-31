# -*- coding: utf-8 -*-
"""
Generate per-pixel uncertainty from classification results based on 
multivariate Gaussian models.

Not filtering with a land mask.

Kristian Hindberg @ CIRFA/UiT, January 2023
"""
import os
os.chdir(f'C:/Users/{os.getlogin()}/OneDrive - UiT Office 365/Uncertainty/')
from S1_uncertainty_2023 import uncertainty, Mahalanobis_distance
from classification_KH import full_S1_image_processing_chain as S1_full
# Do notice that I am loaded my edited classifcation.py file above here.


main_img_folder = 'D:/SENTINEL/Johannes/'
img_ID = 'S1A_EW_GRDM_1SDH_20220503T082725_20220503T082825_043044_0523D1_DCC4'

safe_folder = f'{main_img_folder}{img_ID}/{img_ID}.SAFE/'
feat_folder = safe_folder
result_folder = f'{main_img_folder}{img_ID}/'

fn_classifiers = f'C:/Users/{os.getlogin()}/OneDrive - UiT Office 365/CIRFA/sea_ice_classification/src/classification/classifier/'
classifier_name = 'belgica_bank_classifier_4_classes_20220421'
classifier_path = f'{fn_classifiers}{classifier_name}.pickle'

# Run full processing chain
probs_out, Mahal_out, shape = S1_full(safe_folder = safe_folder, feat_folder = feat_folder, 
                                      result_folder = result_folder, classifier_path = classifier_path,
                                      valid = False, overwrite = True)

# Call function that estimates uncertainty 
n_feat = 2
uncertainty_apost, uncertainty_Mahal = uncertainty(probs_img = probs_out, Mahal_img = Mahal_out, n_feat = n_feat)


# Write uncertainty estimates to file in image format 
import numpy as np
from osgeo import gdal
# Reshape to image geometry when writing to files below 

# Extract number of pixels in x and y direction
Nx, Ny = shape

fn_apost     = f'{main_img_folder}{img_ID}/{img_ID}_apost_uncertainty.img'
# Get drivers (second/third inputs are number of pixels in x/y direction)
output_apost  = gdal.GetDriverByName('Envi').Create(fn_apost, Ny, Nx, 1, gdal.GDT_Float32)
output_apost.GetRasterBand(1).WriteArray(np.reshape(uncertainty_apost,shape))
output_apost.FlushCache()  

fn_Mahal     = f'{main_img_folder}{img_ID}/{img_ID}_Mahal_uncertainty.img'
# Get drivers (second/third inputs are number of pixels in x/y direction)
output_Mahal  = gdal.GetDriverByName('Envi').Create(fn_Mahal, Ny, Nx, 1, gdal.GDT_Float32)
# Write labels and apost of class 1 to band 1 and the apost of each class to different bands
output_Mahal.GetRasterBand(1).WriteArray(np.reshape(uncertainty_Mahal,shape))
# Flush
output_Mahal.FlushCache()
    
