import ice_type_classification.classification as cl
import matplotlib.pyplot as plt
from osgeo import gdal


f_base = 'S1A_EW_GRDM_1SDH_20220502T074631_20220502T074731_043029_05233F_CC06'

feat_folder     = f'data/features/{f_base}'
result_folder   = f'data/results/{f_base}'
clf_pickle_file ='src/ice_type_classification/clf_models/belgica_bank_classifier_4_classes_20220421.pickle'

loglevel  = 'DEBUG'
overwrite = True


cl.classify_S1_image_from_feature_folder(feat_folder, result_folder, clf_pickle_file, loglevel=loglevel, overwrite=overwrite)




















HH = gdal.Open(f'{feat_folder}/Sigma0_HH_db.img').ReadAsArray()
HV = gdal.Open(f'{feat_folder}/Sigma0_HV_db.img').ReadAsArray()
IA = gdal.Open(f'{feat_folder}/IA.img').ReadAsArray()

labels = gdal.Open(f'{result_folder}/{f_base}_labels.img').ReadAsArray()


fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
axes = axes.ravel()
axes[0].imshow(HH, vmin=-35, vmax=0, cmap='gray')
axes[1].imshow(HV, vmin=-40, vmax=-5, cmap='gray')
axes[2].imshow(labels, interpolation='nearest')

axes[0].set_title=('HH')
axes[1].set_title=('HV')
axes[2].set_title=('labels')
