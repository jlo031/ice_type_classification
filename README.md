# Ice type classification
This library provides code for supervised classification of sea ice types in SAR imagery, using the _GIA_ classifier, which accounts for per-class variation of backscatter intensity with incident angle.
Further explanation and theoretical background of the GIA classifer can be found in:
- [Lohse et al (2020)]
- [Lohse et al (2021)]


### Conda environment
The Geospatial Data Abstraction Layer ([GDAL]) library is required to run the code.
The simplest way to use GDAL with Python is to get the Anaconda Python distribution.
It is recommended to run the code in a virtual environment.

    # create new environment
    conda create -y -n ice_types gdal
    
    # activate environment
    conda activate ice_types
    
    # install required packages
    conda install -y ipython scipy loguru
    pip install -U scikit-learn


### Installation

Clone the repository:

    # clone the repository
    git clone git@github.com:jlo031/ice_type_classification.git

Change into the main directory of the cloned repository (it should contain the '_setup.py_' file) and install the library:

    # installation
    pip install .


### Usage

Easy usage examples are provided in the "examples" folder.


[GDAL]: https://gdal.org/
[Lohse et al (2020)]: https://www.researchgate.net/publication/342396165_Mapping_sea-ice_types_from_Sentinel-1_considering_the_surface-type_dependent_effect_of_incidence_angle
[Lohse et al (2021)]: https://www.researchgate.net/publication/349055291_Incident_Angle_Dependence_of_Sentinel-1_Texture_Features_for_Sea_Ice_Classification
