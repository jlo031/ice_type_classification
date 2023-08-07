import S1_processing.S1_feature_extraction as S1_feat
import ice_type_classification.classification as cl

import matplotlib.pyplot as plt
from osgeo import gdal

import pathlib

f_base = 'S1A_EW_GRDM_1SDH_20220502T074527_20220502T074631_043029_05233F_7BC7'

ML = '9x9'

S1_folder = pathlib.Path('/media/Data/Sentinel-1')

L1_folder       = S1_folder / 'L1'
safe_folder     = L1_folder / f'{f_base}.SAFE'
feat_folder     = S1_folder / f'ML_{ML}' / 'features' / f'{f_base}'
result_folder   = S1_folder / f'ML_{ML}' / 'results' / f'{f_base}'

clf_pickle_file = f'/home/jo/work/ice_type_classification/src/ice_type_classification/clf_models/belgica_bank_classifier_4_classes_20220421.pickle'

loglevel  = 'DEBUG'
overwrite = True



# ------------------------------------------------------------------------------ #

# get HH and HV
for intensity in ['HH', 'HV']:
    S1_feat.get_S1_intensity(
        safe_folder,
        feat_folder,
        intensity,
        ML=ML,
        dB=True,
        overwrite=False,
        dry_run=False,
        loglevel=loglevel,
    )

# get IA
S1_feat.get_S1_IA(
    safe_folder,
    feat_folder,
    overwrite=False,
    dry_run=False,
    loglevel=loglevel,
)

# ------------------------------------------------------------------------------ #

# classify

cl.classify_S1_image_from_feature_folder(
    feat_folder,
    result_folder,
    clf_pickle_file,
    uncertainties=True,
    loglevel=loglevel,
    overwrite=overwrite
)

# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

make_figure = True

if make_figure:
    HH = gdal.Open((feat_folder/f'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV = gdal.Open((feat_folder/f'Sigma0_HV_db.img').as_posix()).ReadAsArray()
    IA = gdal.Open((feat_folder/f'IA.img').as_posix()).ReadAsArray()
    labels = gdal.Open((result_folder/f'{f_base}_labels.img').as_posix()).ReadAsArray()
    mahal_uncertainty = gdal.Open((result_folder/f'{f_base}_mahal_uncertainty.img').as_posix()).ReadAsArray()
    apost_uncertainty = gdal.Open((result_folder/f'{f_base}_apost_uncertainty.img').as_posix()).ReadAsArray()


    sub = 3

    fig, axes = plt.subplots(2,3, sharex=True, sharey=True, figsize=((12,10)))
    axes = axes.ravel()
    axes[0].imshow(HH[::sub,::sub], vmin=-35, vmax=0, cmap='gray')
    axes[1].imshow(HV[::sub,::sub], vmin=-40, vmax=-5, cmap='gray')
    axes[2].imshow(labels[::sub,::sub], interpolation='nearest')

    axes[3].imshow(mahal_uncertainty[::sub,::sub], interpolation='nearest')
    axes[4].imshow(apost_uncertainty[::sub,::sub], interpolation='nearest')

    axes[0].set_title('HH')
    axes[1].set_title('HV')
    axes[2].set_title('labels')
    axes[3].set_title('mahal_uncertainty')
    axes[4].set_title('apost_uncertainty')


