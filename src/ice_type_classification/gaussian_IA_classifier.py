"""
---- This is <gaussian_IA_classifier.py> ----

Multi-dimensional Gaussian classifier with and without per-class IA variation
"""

from loguru import logger

import pickle

import numpy as np

from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class gaussian_IA_clf:

    def __init__(self, IA_0=30, override_slopes=False):
        self.info = dict()
        self.info['type']            = 'gaussian_IA'
        self.info['IA_0']            = IA_0
        self.info['override_slopes'] = override_slopes
        self.IA_0                    = IA_0
        self.override_slopes         = override_slopes

# ---------------- #

    def add_info(self, info_dict):
        """Add info_dict to classifier object info

        :param info_dict: dict with classifier information.
        """

        self.info.update(info_dict)

# ---------------- #

    def fit(self, X_train, y_train, IA_train, n_class=[]):
        """Estimate class-dependent slopes for each dimension in X_train vs IA

        :param X_train: training data [N,d]
        :param y_train: training labels or [N,]
        :param IA_train: training incidence angle [N,1] or [N,]
        :param n_class: number of classes
        :return self.n_feat: number of features
        :return self.n_class: number of classes
        :return self.trained_classes: list of actually trained classes
        :return self.a: slope intercept at IA=0
        :return self.b: slope covariance matrix for each class [n_class,d,d]
        """

        ##from sklearn.linear_model import LinearRegression


        # assert correct dimensionality of X_train
        assert len(X_train.shape) == 2, \
            "Training data must be of shape (N,d)."

        # get number of training points and number of features
        N, self.n_feat = X_train.shape
        logger.debug('Number of training points: %i' % N)
        logger.debug('Number of features:        %i' % self.n_feat)

        # find all classes in training data
        classes = np.unique(y_train).astype(int)

        # set maximum number of classes
        if n_class == []:
            self.n_class = classes.max()
            logger.debug('Setting number of classes to %i.' % self.n_class)
        else:
            self.n_class = n_class
            logger.debug('Number of classes given as %i.' % self.n_class)

        logger.debug('Total number of classes:   %i' % self.n_class)
        logger.debug('Classes in training data:  %s' % str(classes))
  

        # initialise means and covariances as nan for all classes
        self.a     = np.zeros((self.n_class, self.n_feat))
        self.b     = np.zeros((self.n_class, self.n_feat))
        self.a.fill(np.nan)
        self.b.fill(np.nan)

        # initialise projected training data X_p
        X_p = np.zeros(X_train.shape)


        # loop over classes in training data
        for i, cl in enumerate(classes):

            logger.debug('Processing class %i.' % cl)

            # loop over all dimensions
            for feat in range(self.n_feat):

                logger.debug('Estimating a and b for class %i and feature %i.' % (cl, feat))

                # fit a model and do the regression
                model = LinearRegression()
                model.fit(np.reshape(IA_train[y_train==cl],(-1,1)), np.reshape(X_train[y_train==cl,feat],(-1,1)))

                # extract intercept and slope
                self.a[cl-1,feat] = model.intercept_
                self.b[cl-1,feat] = model.coef_

                # project current dimensionof X values to IA_0
                X_p[y_train==cl,feat] = X_train[y_train==cl,feat] - self.b[cl-1,feat] * (IA_train[y_train==cl]-self.IA_0)

        if self.override_slopes:
            ## TEMPORARY: OVERRIDE SOME SLOPES MANUALLY
            self.b[1,:] = 0.0
            self.b[2,:] = 0.0
            self.b[4,0] = -0.05
            self.b[4,1] = 0.0
            self.b[5,1] = 0.0
            self.b[5,0] = -0.05

            self.b[6,1] = -0.00
            self.b[7,1] = -0.00

        logger.debug('Estimated slope and intercept for all classes in training data.')
        logger.debug('Projected X values to IA_0.')


        # initialise means and covariances
        self.mu    = np.zeros((self.n_class, self.n_feat))
        self.Sigma = np.zeros((self.n_class, self.n_feat,self.n_feat))
        self.mu.fill(np.nan)
        self.Sigma.fill(np.nan)

        # initialise multivariate gaussians as dict
        self.class_mvn = dict()

        # initialise list of trained classes
        self.trained_classes = []


        # loop over classes in training set
        for cl in classes:

            logger.debug('Estimating mu and Sigma for class %i.' % cl)

            # add current class to list of trained classes
            self.trained_classes.append(cl)

            # select training for current class
            X_cl = X_p[y_train==cl,:]

            logger.debug('Number of training points for current class: %i' % X_cl.shape[0])
            logger.debug('Number of dimensions for current class:      %i' % X_cl.shape[1])

            # find mu and Sigma
            self.mu[cl-1,:]      = X_cl.mean(0)
            self.Sigma[cl-1,:,:] = np.cov(np.transpose(X_cl))

            # initialise distribution
            self.class_mvn[str(cl)] = mvn(self.mu[cl-1,:],self.Sigma[cl-1,:,:])

        return



    """
    IA_step = 1.
    IA_step_half = IA_step/2

    # linearly equal spaced IA vector over full range
    IA_lin = np.arange(20,46,IA_step)

    # corresponding vector for actual IA mean value
    IA_lin_mean = np.zeros(IA_lin.shape)
    IA_lin_mean[:] = np.nan

    # corresponding vector for current feature
    F_lin = np.zeros(IA_lin.shape)
    F_lin[:] = np.nan


    for ii, current_IA in enumerate(IA_lin):

        print(' ')
        print(ii)
        print(current_IA)

        current_idx = np.where((IA_old>current_IA-IA_step_half) & (IA_old<current_IA+IA_step_half))[0]


        F_lin[ii]   = HH_old[current_idx].mean()
        IA_lin_mean = IA_old[current_idx].mean()

        print(current_idx.shape)
        print(F_lin[ii])
        print(IA_lin[ii])



    model = LinearRegression()

    model.fit(np.reshape(IA_lin,(-1,1)), np.reshape(F_lin,(-1,1)))
    """


# ---------------- #

    def predict(self, X_test, IA_test):
        """Predict class labels y_pred for input data X_test with IA_test

        :param X_test: test data [N,d]
        :param IA_test: training incidence angle [N,1] or [N,]
        :return y_pred: predicted class label [N,]
        :return p: probabilities [N,n_class]
        """

        # assert correct dimensionality of X_test
        assert len(X_test.shape) == 2, "Test data must be of shape (N,d)."

        # get number of test points and dimensionality
        N_test, d_test = X_test.shape
        logger.debug('Number of test points: %i.' % N_test)

        # assert correct number of features
        assert d_test == self.n_feat, \
            "Classifier is trained for %i features. " % self.n_feat + \
            "X_test has %i features." % d_test


        # initialise labels and probabilities
        p      = np.zeros((N_test, self.n_class))
        y_pred = np.zeros(N_test)
        p.fill(np.nan)
        y_pred.fill(np.nan)


        # estimate p for trained class
        for cl in (self.trained_classes):

            logger.debug('Working on class %i.' % cl)
            logger.debug('Projecting data according to current class slopes.')

            # initialise projected X_test_p
            X_test_p = np.zeros(X_test.shape)

            # correct X according to class-dependent slope
            for feat in range(self.n_feat):
                X_test_p[:,feat] = X_test[:,feat] - self.b[cl-1,feat] * (IA_test-self.IA_0)

                logger.debug('Calculating p for class %i.' % cl)

            # estimate probability from multivariate gaussian function.
            p[:,cl-1] = self.class_mvn[str(cl)].pdf(X_test_p)

        # find maximum p and set labels
        y_pred  = np.nanargmax(p,1) + 1

        return y_pred, p

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class gaussian_clf:

    def __init__(self):
        self.info = dict()
        self.info['type'] = 'gaussian'

# ---------------- #

    def add_info(self, info_dict):
        """Add info_dict to classifier object info.

        :param info_dict: dict with classifier information.
        """

        self.info.update(info_dict)

# ---------------- #

    def fit(self, X_train, y_train, n_class=[]):
        """Fit classifier to input training data X_train with labels y_train.

        :param X_train: training data [N,d]
        :param y_train: training labels [N,1] or [N,]
        :param n_class: number of classes
        :return self.n_feat: number of features
        :return self.n_class: number of classes
        :return self.trained_classes: list of actually trained classes
        :return self.mu: mean vector for each class [n_class,d]
        :return self.Sigma: covariance matrix for each class [n_class,d,d]
        """

        # assert correct dimensionality of X_train
        assert len(X_train.shape) == 2, "Training data must be of shape (N,d)."

        # get number of training points and number of features
        N, self.n_feat = X_train.shape
        logger.debug('Number of training points: %i' % N)
        logger.debug('Number of features:        %i' % self.n_feat)

        # find all classes in training data
        classes = np.unique(y_train).astype(int)

        # set maximum number of classes
        if n_class == []:
            self.n_class = classes.max()
            logger.debug('Setting number of classes to %i.' % self.n_class)
        else:
            self.n_class = n_class
            logger.debug('Number of classes given as %i.' % self.n_class)

        logger.debug('Total number of classes:   %i' % self.n_class)
        logger.debug('Classes in training data:  %s' % str(classes))


        # initialise means and covariance matrices as nan for all classes
        self.mu    = np.zeros((self.n_class,self.n_feat))
        self.Sigma = np.zeros((self.n_class,self.n_feat,self.n_feat))
        self.mu.fill(np.nan)
        self.Sigma.fill(np.nan)

        # initialise multivariate gaussians as dict
        self.class_mvn = dict()

        # initialise list of trained classes
        self.trained_classes = []


        # loop over classes in training data
        for cl in classes:

            logger.debug('Estimating mu and Sigma for class %i.' % cl)

            # add current class to list of trained classes
            self.trained_classes.append(cl)

            # select training for current class
            X_cl = X_train[y_train==cl,:]

            logger.debug('Number of training points for current class: %i' % X_cl.shape[0] )
            logger.debug('Number of dimensions for current class: %i' % X_cl.shape[1] )

            # find mu and Sigma
            self.mu[cl-1,:]      = X_cl.mean(0)
            self.Sigma[cl-1,:,:] = np.cov(np.transpose(X_cl))

            # initialise distribution
            self.class_mvn[str(cl)] = mvn(self.mu[cl-1,:],self.Sigma[cl-1,:,:])

        return

# ---------------- #

    def predict(self, X_test):
        """Predict class labels y_pred for input data X_test.

        :param X_test: test data [N,d]
        :return y_pred: predicted class label [N,]
        :return p: probabilities [N,n_class]
        """

        # assert correct dimensionality of X_test
        assert len(X_test.shape) == 2, "Test data must be of shape (N,d)."

        # get number of test points and dimensionality
        N_test, d_test = X_test.shape
        logger.debug('Number of test points: %i.' % N_test)

        # assert correct number of features
        assert d_test == self.n_feat, \
            "Classifier is trained for %i features. " % self.n_feat + \
            "X_test has %i features." % d_test


        # initialise labels and probabilities
        p      = np.zeros((N_test, self.n_class))
        y_pred = np.zeros(N_test)
        p.fill(np.nan)
        y_pred.fill(np.nan)


        # estimate p for trained class
        for cl in (self.trained_classes):

            logger.debug('Calculating p for class %i.' % cl)

            # estimate probability from multivariate gaussian function.
            p[:,cl-1] = self.class_mvn[str(cl)].pdf(X_test)

        # find maximum p and set labels
        y_pred  = np.nanargmax(p,1) + 1

        return y_pred, p

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def get_per_class_score(y_true, y_pred, average=True, subsample=1):
    """Calculate average per class classification accuracy.

    :param y_true: True class labels [N,]
    :param y_pred: Predicted class labels [N]
    :param average: True/False for average per-class CA or per-class CA
                    (default True)
    :param subsample: Subsample labels (default=1)
    :return CA: Classification accuracy (per class or average per class)
    """

    # convert labels to array
    y_true = np.array(y_true)[::subsample]
    y_pred = np.array(y_pred)[::subsample]

    assert y_true.shape == y_pred.shape , 'true labels and predicted labels must have the same shape.'

    classes = set(y_true)
    CA = np.array(list(np.equal(y_true[y_true==label], y_pred[y_true==label]).sum() / float(sum(y_true==label)) for label in classes))

    if average:
        CA = np.mean(CA)

    return CA

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def write_classifier_dict_2_pickle(output_file, classifier_dict):
    """Write classifier dictionary to pickle file.

    :param output_file: pickle output file
    :param classifier_dict: classifier dictionary
    """
    with open(output_file,'wb') as f: 
        pickle.dump(classifier_dict,f)

    return

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def read_classifier_dict_from_pickle(input_file):
    """Read classifier dictionary from pickle file.

    :param input_file: pickle input file
    :return classifier_dict: classifier dictionary
    """

    with open(input_file,'rb') as f: 
        classifier_dictionary = pickle.load(f)

    return classifier_dictionary

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_gaussian_IA_clf_object_from_params_dict(gaussian_IA_params_dict):
    """Make a gaussian_IA_classifier object from parameters in input dict.

    :param  gaussian_IA_params_dict: dictionary with with classifier parameters
    :return clf: classifier object (gaussian_IA_clf)
     """

    clf                 = gaussian_IA_clf()
    clf.a               = gaussian_IA_params_dict['a']
    clf.b               = gaussian_IA_params_dict['b']
    clf.mu              = gaussian_IA_params_dict['mu']
    clf.Sigma           = gaussian_IA_params_dict['Sigma']
    clf.IA_0            = gaussian_IA_params_dict['IA_0']
    clf.n_class         = gaussian_IA_params_dict['n_class']
    clf.n_feat          = gaussian_IA_params_dict['n_feat']
    clf.trained_classes = gaussian_IA_params_dict['trained_classes']

    # create the mvn objects for the installed scipy version
    # mvn requires mu and Sigma

    clf.class_mvn = dict()

    for cl in clf.trained_classes:
        clf.class_mvn[str(cl)] = mvn(clf.mu[cl-1,:],clf.Sigma[cl-1,:,:])

    return clf

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_gaussian_IA_params_dict_from_clf_object(clf):
    """Make a gaussian_IA_params dict from a gaussian_IA_classifier object.

    :param clf: classifier object (gaussian_IA_clf)
    :return gaussian_IA_params_dict: dictionary with with classifier parameters
     """
    # initialize dict for clf parameters
    gaussian_IA_params_dict = dict()

    # fill in dict with clf parameters
    gaussian_IA_params_dict['a']               = clf.a
    gaussian_IA_params_dict['b']               = clf.b
    gaussian_IA_params_dict['mu']              = clf.mu
    gaussian_IA_params_dict['Sigma']           = clf.Sigma
    gaussian_IA_params_dict['IA_0']            = clf.IA_0
    gaussian_IA_params_dict['n_class']         = clf.n_class
    gaussian_IA_params_dict['n_feat']          = clf.n_feat
    gaussian_IA_params_dict['trained_classes'] = clf.trained_classes

    return gaussian_IA_params_dict

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_gaussian_params_dict_from_clf_object(clf):
    """Make a gaussian_params dict from a gaussian_classifier object.

    :param clf: classifier object (gaussian_clf)
    :return gaussian_params_dict: dictionary with with classifier parameters
     """
    # initialize dict for clf parameters
    gaussian_params_dict = dict()

    # fill in dict with clf parameters
    gaussian_params_dict['mu']              = clf.mu
    gaussian_params_dict['Sigma']           = clf.Sigma
    gaussian_params_dict['n_class']         = clf.n_class
    gaussian_params_dict['n_feat']          = clf.n_feat
    gaussian_params_dict['trained_classes'] = clf.trained_classes

    return gaussian_params_dict

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_gaussian_clf_object_from_params_dict(gaussian_params_dict):
    """Make a gaussian_classifier object from parameters in input dict.

    :param  gaussian_params_dict: dictionary with with classifier parameters
    :return clf: classifier object (gaussian_clf)
     """

    clf                 = gaussian_clf()

    clf.mu              = gaussian_params_dict['mu']
    clf.Sigma           = gaussian_params_dict['Sigma']
    clf.n_class         = gaussian_params_dict['n_class']
    clf.n_feat          = gaussian_params_dict['n_feat']
    clf.trained_classes = gaussian_params_dict['trained_classes']


    # create the mvn objects for the installed scipy version
    # mvn requires mu and Sigma

    clf.class_mvn = dict()

    for cl in clf.trained_classes:
        clf.class_mvn[str(cl)] = mvn(clf.mu[cl-1,:],clf.Sigma[cl-1,:,:])

    # OLD, delete if the other stuff works
    # create the mvn objects for the installed scipy version
    ##clf.class_mvn       = gaussian_params_dict['class_mvn']

    return clf

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <gaussian_IA_classifier.py> ----
