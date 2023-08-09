# ---- This is <uncertainty_utils.py> ----

"""
Module with functions needed for uncertainty estimation in statistical clf
"""

import argparse
import os
import sys
import pathlib
import shutil

from loguru import logger

import numpy as np
from scipy.stats import chi2

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def get_mahalanobis_distance(
    X_test,
    mu_vec,
    cov_mat,
    IA_test = False,
    IA_0 = np.nan,
    IA_slope = np.nan,
    loglevel = 'INFO',
):

    """Find Mahalanobis distance (not squared) for a set of feature vectors given fixed class parameters (mean vector and covariance matrix)
       
    Parameters
    ----------
    X_test : numpy array of N samples of dimension d [N, d]
    mu_vec : numpy array of mean values of each of C classes over d dimensions [C, d]
    cov_mat : numpy array of covariance matrices of C classes over d dimensions [C, d, d]
    IA_test : numpy arry of N samples with IA values (for models with linearly variable mean vector)
    IA_0: IA test value (for models with linearly variable mean vector)
    IA_slope : numoy array of C classes and d dimensions with slope values [C, d](for models with linearly variable mean vector)
    loglevel : loglevel setting (default='INFO')

    Returns
    -------
    mahal_test : Numpy array of N samples and C classes, with Mahalanobis distance of all samples for all classes [N x C]
    """ 

    # get number of classes
    n_classes = mu_vec.shape[0]

    logger.debug(f'Input data has {n_classes} classes')

    # define empty output matrix
    mahal_test  = np.empty((X_test.shape[0],mu_vec.shape[0]))

    # loop over all classes and calculate Mahalanobis distances
    for cl in range(n_classes):

        # extract mean vector and covariance matrix of current class
        mu_vec_curr  = mu_vec[cl]
        cov_mat_curr = cov_mat[cl]

        # initialise projected X_test_p (needed if mean vector has a slope)
        X_test_p = np.zeros(X_test.shape)

        # correct X according to class-dependent slope
        if ( (IA_test[0] == False) == False):
            for feat in range(mu_vec.shape[1]):
              X_test_p[:,feat] = X_test[:,feat] - IA_slope[cl,feat] * (IA_test-IA_0)
        else:
            X_test_p = X_test
            
        # center feature vectors wrt given mean vector        
        X_centered = X_test_p - mu_vec_curr # TODO: should this be X_test_p ?

        # calculated "left" part of the Mahalanobis distance
        left = np.dot(X_centered, np.linalg.inv(cov_mat_curr))

        # do the "right" part so that it handles very large matrices, by 
        # avoiding to construct an [N,N] matrix and taking the trace of this.
        # Take sqrt to get Mahalanobis distance
        mahal_test[:,cl] = np.sqrt(np.sum(np.multiply(left,X_centered), 1))

    return mahal_test

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def uncertainty(
    probs_img,
    Mahal_img,
    n_feat,  
    Aposterior_uncertainty_meausure = "Entropy",
    DO_aposterior_uncertainty = True,
    DO_Mahalanobis_uncertainty = True, 
    Discrete_uncertainty = False
):

    """
    Estimate per-pixel uncertainty arrays based on aposterior probabilities and Mahalanobis distance.
    * Uncertainty based on aposterior probabilities is high when a sample has 
      a feature vector in a location with high class overlap.
    * Uncertainty based on Mahalnobis distance is high when a sample is really
      far from the center of the ellipsoide that defines the multivariate
      Gaussian of the most probable class. A really high Mahalanobis distance
      might be an indication of the sample belonging to a class not in the
      training set, or at least not a class that the classifier is trained on.
      The p-values of the Mahalnobis distance of the most probable class is 
      used to assign this "out-of-distribution uncertainty".
    
    Required input variables:
      - probs_img - [N,C] Array of probabilities of C classes of N samples
      - Mahal_img - [N,C] Mahalanobis distance C classes of N samples
      - n_feat    - Number of features used for classification.  
                    Used to find p-values of Mahalanobis distances.

    Optional inputs:
      Aposterior_uncertainty_meausure 
      - Which method for uncertainty evaluation based on the aposterior 
        probabilities of each class. Possible values are 
            'Entropy'   - Entropy across all clasese (DEFAULT)
            'Apost_max' - Only used aposterior of most probable class
            'Diff_1_2'  - Difference of the two most probable classes
        The methods produce identical results when only using two classes.
      DO_aposterior_uncertainty [Default: True]
      - Bolean to make uncertainty matrix based on posterior values
      DO_Mahalanobis_uncertainty [Default: True]
      - Bolean to make uncertainty matrix based on Mahalanobis distance     
      Discrete_uncertainty
      - Bolean to discretize uncertainty to a fixed set of four levels 
        [Default: False]

      DO_geocode  [Default: True]
      - Bolean to output geocoded results
      res_geocode [default: 40]
      - Pixel spacing in meter (both direction) in geocoded results
      epsg [Default]: 3996]
      - Map projection to use

    Outputs
    - Uncertainty image based on the chosen aposterior uncertainty measure
    - Uncertainty image based on Mahalanobis distance
    
    Kristian Hindberg @ CIRFA/UiT, January 2023
    """

    N, n_classes = probs_img.shape
    
    """
        Uncertainty based on aposterior probabilities of different classes
    """
    if DO_aposterior_uncertainty:

        # Valid mask can be deduced from the pixels that are NaN in probs_img
        valid_mask = np.ones(N).astype(int)
        valid_mask[np.isnan(probs_img[:,0])] = 0

        # Find sum of pdf values for each pixels across all possible classes
        prob_sum = np.sum(probs_img, 1)
        # pixels with pdf values rounded to zero for both clases,
        # which causes errors in the division operation below. 
        # Set pixels with zero prob for all classes to invalid as no estimate
        # of max probability can be found.
        valid_mask[prob_sum == 0] = 0
    
        # Find max aposterior probability of most probable class, with values
        #    nan      For pixels set to invalid in the valid_mask
        #      0      For pixels with so extreme measurement values so that 
        #             the probability is rounded to zero for all classes
        #      <0,1]  Aposterior probability of most probable class 
        apost_all_img = np.ones((N, n_classes))
        apost_all_img.fill(np.nan)
        # First, find sum of pdf values across all classes
        prob_sum_each_pixel = np.nansum(probs_img[valid_mask==1], 1) 
        # Divied each class value by this summed value, so they sum to 1
        for classy in range(n_classes):
            apost_all_img[valid_mask==1, classy] = probs_img[valid_mask==1,classy] / prob_sum_each_pixel
        # Find max aposterior of each pixel 
        apost_max_img = apost_all_img.max(axis=1)
        # Set pixels with zero prob for all classes to zero apost max
        apost_max_img[prob_sum == 0] = 0
        
        
        """
        Calculate uncertainty values for given uncertainty method.
        All methods are scaled to [0,1] with low values indicating 
        low uncertainty in the classification.
        """
        if Aposterior_uncertainty_meausure == "Entropy":            
            # Define entropy-based uncertainty measure, which is based on aposterior 
            # probabilities of all classes.
            # Initially set all to zero to avoid problems in the summation. At the end,
            # set invalid pixels to NaN
            #   [0,1]  Entropy based uncertainty estimate
            #    nan      For pixels set to invalid in the valid_mask
            #      0      For pixels with so extreme measurement values so that the classifier rounds the 
            #             the probability to zero for all but one classes.
            #      1      For pixels with so extreme measurement values so that the classifier rounds the 
            #             the probability to zero for all classes
            #      <0,1>  Aposterior probability of most probable class 
            uncer_entropy = np.zeros(N)
            for classy in range(n_classes):
                
                # Make a copy of the valid mask which will be update for each class
                valid_mask_class = np.copy(valid_mask)
                valid_mask_class[apost_all_img[:,classy] == 0]   = 0
                # Find entropy values for the current class
                entropy_curr_class = - np.multiply(apost_all_img[valid_mask_class==1,classy] , np.log2(apost_all_img[valid_mask_class==1,classy]))
                # Add them to the final entropy
                uncer_entropy[valid_mask_class==1] = uncer_entropy[valid_mask_class==1] + entropy_curr_class
            
            # Set invalid pixels to NaN
            uncer_entropy[valid_mask==0] = np.nan

            # Scale it to [0,1]
            uncer_entropy = uncer_entropy / np.log2(n_classes)
            uncer_entropy[uncer_entropy>1] = 1 # From round-off, you might get some values just above one
              
            # Set those with aposterior of most probable class equal to 1 to zero 
            # entropy based uncertainty. From round-off and numerical issues,
            # even with apost_max_img==1 some pixels will get a very small, but
            # non-zero, positive value for entropy    
            uncer_entropy[apost_max_img==1] = 0    
            uncer_entropy[apost_max_img==0] = 1
        
            # Assign threshold values for discretization of uncertainty
            apost_uncertainty_thresholds = [0.469, 0.722, 0.881, 0.971]
            
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncer_entropy
            
        if Aposterior_uncertainty_meausure == "Apost_max":
            
            # Reasonable cut-off might depend on number of classes
            uncer_apost_max  = 1 - apost_max_img
            
            # Set pixels with all classes rounded to zero prob to lower possible value
            uncer_apost_max[prob_sum == 0] = 1-1/n_classes

            # Scale so that the uncertainty can take values in [0,1]
            uncer_apost_max = uncer_apost_max/(1-1/n_classes)
            
            # Assign threshold values for discretization of uncertainty
            apost_uncertainty_thresholds = [0.2, 0.4, 0.6, 0.8]
            
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncer_apost_max

        if Aposterior_uncertainty_meausure == "Diff_1_2":     

            # Reasonable cut-off might depend on number of classes   
            
            # Determine aposterior probability of two most probable classes
            two_most_probable = np.sort(apost_all_img, axis=1)[:, -2:]
            # Take difference of these two
            Diff_1_2 = two_most_probable[:,1]  - two_most_probable[:,0]
            # Do 1 minus this difference to make uncertainty measure,
            # where high values indicate low difference in aposterior 
            # probability between the two most probable classes.
            uncer_diff_1_2 = 1 - Diff_1_2

            # Assign threshold values for discretization of uncertainty
            apost_uncertainty_thresholds =[0.2, 0.4, 0.6, 0.8]
            
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncer_diff_1_2

        
        """ Discretize uncertainty estimates to fixed set of levels """
        if Discrete_uncertainty==True:           
            """
            The assigned uncertainty thresholds produce identical results for 
            two classes for all three possible uncertainty methods, and corresponds 
            to the most probable class having a posterior probability of
            0.9, 0.8, 0.7 and 0,6, respectively for the increasing uncertainty thresholds.
            
            It does not produce identical result for more than two classes,
            and the thresholds should possibly be set different if using 
            (considerably) more than two classes.
            
            For each method, values below the lowest uncertainty threshold are
            set to NaN to avoid the uncertainty map having values for very
            certain pixels.
            """
            uncertainty_apost_levels_img = np.copy(uncertainty_apost)
            for apost_uncertainty_thresholds_curr in apost_uncertainty_thresholds:
                uncertainty_apost_levels_img[ uncertainty_apost > apost_uncertainty_thresholds_curr] = apost_uncertainty_thresholds_curr
            # Set uncertainty pixels below lowest threshold to nan
            uncertainty_apost_levels_img[ uncertainty_apost_levels_img < apost_uncertainty_thresholds[0] ] = np.nan
    
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncertainty_apost_levels_img
            



    """
        Uncertainty based on high Mahalnobis distance (out-of-distribution)
    """
    if DO_Mahalanobis_uncertainty:
        
        # Define uncertainty matrix based on -log10 of the P-value of the
        # Mahalanobis distance of the most probable class.

        # Find minimum Mahalanobis distance of each pixel
        Mahal_of_maxProbClass = Mahal_img.min(axis=1)
        # Take square of this distance 
        MahalSquared_of_maxProbClass = Mahal_of_maxProbClass * Mahal_of_maxProbClass
        # Square of MD is chi-square distributed with 'n_feat' DOFs
        if Discrete_uncertainty==True:
            
            # Assign uncertainty thresholds (corresponding to -log10(P-values) )
            Mahal_threshold = np.array([6, 8, 10, 12])
    
            # Define output variables - set all to nan first
            uncertainty_Mahal = np.zeros(N)
            uncertainty_Mahal.fill(np.nan)
            # Find pixels that are rejected for different p-value thresholds, and
            # assign the value: -log10(Pvalue_threshold)
            # Pixels not rejected by any threshold are set to nan
            for uncer_threshold in Mahal_threshold:
                alpha_curr = 1/np.float_power(10, uncer_threshold)
                Mahal_pvalue_threshold = chi2.ppf(1-alpha_curr, n_feat)
                uncertainty_Mahal[MahalSquared_of_maxProbClass > Mahal_pvalue_threshold] = -np.log10(alpha_curr)

        else:
            
            # Find p-value of each pixel
            P_val_Mahal = 1 - chi2.cdf(MahalSquared_of_maxProbClass, n_feat )
            
            # Set zero P-values to 1e-17
            P_val_Mahal[P_val_Mahal==0] = 1e-17 # Might be thresholded below
            
            # Take -log10
            P_val_Mahal[np.isnan(MahalSquared_of_maxProbClass) == False] = -np.log10(P_val_Mahal[np.isnan(MahalSquared_of_maxProbClass) == False])
            # Threshold to above 6 and below 12
            P_val_Mahal[P_val_Mahal< 6] = np.nan # TODO: make thresholds flexible
            P_val_Mahal[P_val_Mahal>12] = 12
            
            # Assign to matrix
            uncertainty_Mahal = P_val_Mahal

    else:
        uncertainty_Mahal = False
    
    # return uncertainty maps
    return uncertainty_apost, uncertainty_Mahal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <uncertainty_utils.py> ----
