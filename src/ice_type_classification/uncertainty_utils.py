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

    """Find mahalanobis distance (not squared) for a set of feature vectors given fixed class parameters (mean vector and covariance matrix)
       
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
    mahal_test : Numpy array of N samples and C classes, with mahalanobis distance of all samples for all classes [N x C]
    """ 

    # get number of classes
    n_classes = mu_vec.shape[0]

    logger.debug(f'input data has {n_classes} classes')

    # define empty output matrix
    mahal_test  = np.empty((X_test.shape[0],mu_vec.shape[0]))

    # loop over all classes and calculate mahalanobis distances
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

        # calculated "left" part of the mahalanobis distance
        left = np.dot(X_centered, np.linalg.inv(cov_mat_curr))

        # do the "right" part so that it handles very large matrices, by 
        # avoiding to construct an [N,N] matrix and taking the trace of this.
        # Take sqrt to get mahalanobis distance
        mahal_test[:,cl] = np.sqrt(np.sum(np.multiply(left,X_centered), 1))

    return mahal_test

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def uncertainty(
    probs_img,
    mahal_img,
    n_feat,  
    apost_uncertainty_meausure = "Entropy",
    DO_apost_uncertainty = True,
    DO_mahalanobis_uncertainty = True, 
    discrete_uncertainty = False,
    mahal_thresh_min = 6,
    mahal_thresh_max = 12,
    mahal_discrete_thresholds = np.array([6, 8, 10, 12]),
    apost_discrete_thresholds = ['default'],
):

    """
    Estimate per-pixel uncertainty arrays based on apost probabilities and mahalanobis distance.
    * Uncertainty based on apost probabilities is high when a sample has 
      a feature vector in a location with high class overlap.
    * Uncertainty based on mahalnobis distance is high when a sample is really
      far from the center of the ellipsoide that defines the multivariate
      Gaussian of the most probable class. A really high mahalanobis distance
      might be an indication of the sample belonging to a class not in the
      training set, or at least not a class that the classifier is trained on.
      The p-values of the mahalnobis distance of the most probable class is 
      used to assign this "out-of-distribution uncertainty".
    
    Required input variables:
      - probs_img - [N,C] Array of probabilities of C classes of N samples
      - mahal_img - [N,C] mahalanobis distance C classes of N samples
      - n_feat    - Number of features used for classification.  
                    Used to find p-values of mahalanobis distances.

    Optional inputs:
      apost_uncertainty_meausure 
      - Which method for uncertainty evaluation based on the apost 
        probabilities of each class. Possible values are 
            'Entropy'   - Entropy across all clasese (DEFAULT)
            'Apost_max' - Only used apost of most probable class
            'Diff_1_2'  - Difference of the two most probable classes
        The methods produce identical results when only using two classes.
      DO_apost_uncertainty [Default: True]
      - Bolean to make uncertainty matrix based on posterior values
      DO_mahalanobis_uncertainty [Default: True]
      - Bolean to make uncertainty matrix based on mahalanobis distance     
      discrete_uncertainty
      - Bolean to discretize uncertainty to a fixed set of four levels 
        [Default: False]

    mahal_thresh_min : minimum threshold for mahalanobis distance (default=6)
    mahal_thresh_max : maximum threshold for mahalanobis distance (default=12)
    mahal_discrete_thresholds : np.array with thresholds for discretization of mahal uncertainty (default: np.array([6, 8, 10, 12]))
    apost_discrete_thresholds : thresholds for discretization of apost uncertainty (default: depending on uncertainty_measure)

    Outputs
    - Uncertainty image based on the chosen apost uncertainty measure
    - Uncertainty image based on mahalanobis distance
    
    Kristian Hindberg @ CIRFA/UiT, January 2023
    """

    N, n_classes = probs_img.shape

    uncertainty_measure_choices = [
        'Entropy',
        'entropy',
        'Apost_max',
        'apost_max',
        'Diff_1_2',
        'diff_1_2'
    ]

    """
        Uncertainty based on apost probabilities of different classes
    """
    if DO_apost_uncertainty:

        if apost_uncertainty_meausure not in uncertainty_measure_choices:
            logger.error(f'Invalid choice for apost_uncertainty_meausure: {apost_uncertainty_meausure}')
            logger.error(f'Valid choices are: {uncertainty_measure_choices}')
            uncertainty_apost = False


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
    
        # Find max apost probability of most probable class, with values
        #    nan      For pixels set to invalid in the valid_mask
        #      0      For pixels with so extreme measurement values so that 
        #             the probability is rounded to zero for all classes
        #      <0,1]  apost probability of most probable class 
        apost_all_img = np.ones((N, n_classes))
        apost_all_img.fill(np.nan)
        # First, find sum of pdf values across all classes
        prob_sum_each_pixel = np.nansum(probs_img[valid_mask==1], 1) 
        # Divied each class value by this summed value, so they sum to 1
        for classy in range(n_classes):
            apost_all_img[valid_mask==1, classy] = probs_img[valid_mask==1,classy] / prob_sum_each_pixel
        # Find max apost of each pixel 
        apost_max_img = apost_all_img.max(axis=1)
        # Set pixels with zero prob for all classes to zero apost max
        apost_max_img[prob_sum == 0] = 0
        
        
        """
        Calculate uncertainty values for given uncertainty method.
        All methods are scaled to [0,1] with low values indicating 
        low uncertainty in the classification.
        """

        if apost_uncertainty_meausure in ["Entropy", "entropy"]:
            # Define entropy-based uncertainty measure, which is based on apost 
            # probabilities of all classes.
            # Initially set all to zero to avoid problems in the summation. At the end,
            # set invalid pixels to NaN
            #   [0,1]  Entropy based uncertainty estimate
            #    nan      For pixels set to invalid in the valid_mask
            #      0      For pixels with so extreme measurement values so that the classifier rounds the 
            #             the probability to zero for all but one classes.
            #      1      For pixels with so extreme measurement values so that the classifier rounds the 
            #             the probability to zero for all classes
            #      <0,1>  apost probability of most probable class

            logger.debug(f'apost_uncertainty measure is: {apost_uncertainty_meausure}')

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
              
            # Set those with apost of most probable class equal to 1 to zero 
            # entropy based uncertainty. From round-off and numerical issues,
            # even with apost_max_img==1 some pixels will get a very small, but
            # non-zero, positive value for entropy    
            uncer_entropy[apost_max_img==1] = 0    
            uncer_entropy[apost_max_img==0] = 1
        
            # Assign threshold values for discretization of uncertainty
            if apost_discrete_thresholds[0] == 'default':
                apost_uncertainty_thresholds = [0.469, 0.722, 0.881, 0.971]
            else:
                apost_uncertainty_thresholds = apost_discrete_thresholds

            # Assign name used below for uncertainty variable
            uncertainty_apost = uncer_entropy
            
        if apost_uncertainty_meausure in ["Apost_max", "apost_max"]:

            logger.debug(f'apost_uncertainty measure is: {apost_uncertainty_meausure}')

            # Reasonable cut-off might depend on number of classes
            uncer_apost_max  = 1 - apost_max_img
            
            # Set pixels with all classes rounded to zero prob to lower possible value
            uncer_apost_max[prob_sum == 0] = 1-1/n_classes

            # Scale so that the uncertainty can take values in [0,1]
            uncer_apost_max = uncer_apost_max/(1-1/n_classes)
            
            # Assign threshold values for discretization of uncertainty
            if apost_discrete_thresholds[0] == 'default':
                apost_uncertainty_thresholds = [0.2, 0.4, 0.6, 0.8]
            else:
                apost_uncertainty_thresholds = apost_discrete_thresholds
            
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncer_apost_max

        if apost_uncertainty_meausure in ["Diff_1_2", "diff_1_2"]:     

            logger.debug(f'apost_uncertainty measure is: {apost_uncertainty_meausure}')

            # Reasonable cut-off might depend on number of classes   
            
            # Determine apost probability of two most probable classes
            two_most_probable = np.sort(apost_all_img, axis=1)[:, -2:]
            # Take difference of these two
            Diff_1_2 = two_most_probable[:,1]  - two_most_probable[:,0]
            # Do 1 minus this difference to make uncertainty measure,
            # where high values indicate low difference in apost 
            # probability between the two most probable classes.
            uncer_diff_1_2 = 1 - Diff_1_2

            # Assign threshold values for discretization of uncertainty
            if apost_discrete_thresholds[0] == 'default':
                apost_uncertainty_thresholds = [0.2, 0.4, 0.6, 0.8]
            else:
                apost_uncertainty_thresholds = apost_discrete_thresholds
            
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncer_diff_1_2

        
        """ Discretize uncertainty estimates to fixed set of levels """
        if discrete_uncertainty==True:           
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

            logger.debug('discretizing apost_uncertainty')
            logger.debug(f'discretization levels: {apost_uncertainty_thresholds}')

            uncertainty_apost_levels_img = np.copy(uncertainty_apost)

            for apost_uncertainty_thresholds_curr in apost_uncertainty_thresholds:
                uncertainty_apost_levels_img[uncertainty_apost>apost_uncertainty_thresholds_curr] = apost_uncertainty_thresholds_curr
            # Set uncertainty pixels below lowest threshold to nan
            uncertainty_apost_levels_img[uncertainty_apost_levels_img<apost_uncertainty_thresholds[0] ] = np.nan
    
            # Assign name used below for uncertainty variable
            uncertainty_apost = uncertainty_apost_levels_img
            



    """
        Uncertainty based on high mahalnobis distance (out-of-distribution)
    """
    if DO_mahalanobis_uncertainty:
        
        # Define uncertainty matrix based on -log10 of the P-value of the
        # mahalanobis distance of the most probable class.

        # Find minimum mahalanobis distance of each pixel
        mahal_of_maxProbClass = mahal_img.min(axis=1)
        # Take square of this distance 
        mahalSquared_of_maxProbClass = mahal_of_maxProbClass * mahal_of_maxProbClass
        # Square of MD is chi-square distributed with 'n_feat' DOFs
        if discrete_uncertainty==True:
            
            # Assign uncertainty thresholds (corresponding to -log10(P-values) )
            mahal_threshold = mahal_discrete_thresholds
    
            # Define output variables - set all to nan first
            uncertainty_mahal = np.zeros(N)
            uncertainty_mahal.fill(np.nan)
            # Find pixels that are rejected for different p-value thresholds, and
            # assign the value: -log10(Pvalue_threshold)
            # Pixels not rejected by any threshold are set to nan
            for uncer_threshold in mahal_threshold:
                alpha_curr = 1/np.float_power(10, uncer_threshold)
                mahal_pvalue_threshold = chi2.ppf(1-alpha_curr, n_feat)
                uncertainty_mahal[mahalSquared_of_maxProbClass > mahal_pvalue_threshold] = -np.log10(alpha_curr)

        else:
            
            # Find p-value of each pixel
            P_val_mahal = 1 - chi2.cdf(mahalSquared_of_maxProbClass, n_feat )
            
            # Set zero P-values to 1e-17
            P_val_mahal[P_val_mahal==0] = 1e-17 # Might be thresholded below
            
            # Take -log10
            P_val_mahal[np.isnan(mahalSquared_of_maxProbClass) == False] = -np.log10(P_val_mahal[np.isnan(mahalSquared_of_maxProbClass) == False])

            # Threshold to mahal_thresh_max and mahal_thresh_min
            P_val_mahal[P_val_mahal<mahal_thresh_min] = np.nan
            P_val_mahal[P_val_mahal>mahal_thresh_max] = mahal_thresh_max
            
            # Assign to matrix
            uncertainty_mahal = P_val_mahal

    else:
        uncertainty_mahal = False
    

    return uncertainty_apost, uncertainty_mahal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <uncertainty_utils.py> ----
