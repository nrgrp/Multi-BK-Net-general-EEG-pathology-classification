# Multi-BK-Net-general-EEG-pathology-classification
Code accompanies the paper [Multi-BK-Net: Multi-Branch Multi-Kernel Convolutional Neural Networks for Clinical EEG Analysis](https://openreview.net/forum?id=IsG10xZAaA).



## Abstract



## Contents

Temple University Hospital (TUH) Abnormal EEG Corpus (TUAB):

TUAB train/test split as provided by the TUH.

TUH Abnormal Expansion Balanced EEG Corpus (TUABEXB):

TUABEXB train/test split as described in [Kiessner et al. (2023)](https://www.sciencedirect.com/science/article/pii/S2213158223001730).


TUH Abnormal Combined EEG Corpus (TUABCOMB): combination of TUAB train set and TUABEXB train set

- dataset_description/ - description of all datasets and train/test data splits. Each description contains the path of the recordings used and pathology labels based on automated classification of the medical text reports. 'path_new' refers to the same recording in the current version of the Temple University Hospital EEG Corpus (v2.0.0).

- code/ - Python scripts and notebooks for data preprocessing, model optimisation and model training and evaluation.

## Environments

This repository expects a working installation of [braindecode](https://github.com/braindecode/braindecode).  
Additionally, it requires to install packages listed in `environment.yml`. So download the reposiory and create the environment from the environment.yml file:

```
conda env create -f environment.yml
```

The requirements.txt file lists all Python libraries that the scripts and notebooks depend on, and they can be installed using:

```
pip install -r requirements.txt
```


## Data

Our study is based on the Temple University Hospital Abnormal EEG Corpus (v2.0.0) and the TUH Abnormal Expansion Balanced EEG Corpus (TUABEXB). Both datasets are subsets of the TUH EEG Corpus (v1.1.0 and v1.2.0) avilable for download at: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml. Code curating the TUABEXB [TUHAbnormal-Expansion-dataset](https://github.com/AKiessner/TUHAbnormal-Expansion-dataset).


## Citing

If you use this code in a scientific publication, please cite us as:

```
@article{kiessner2025MultiBKNet, 
  title = {Multi-BK-Net: Multi-Branch Multi-Kernel Convolutional Neural Networks for Clinical EEG Analysis},
  journal = {Transactions on Machine Learning Research},
  url = {https://www.sciencedirect.com/science/article/pii/S2213158223001730](https://openreview.net/forum?id=IsG10xZAaA},
  author = {Ann-Kathrin Kiessner and Joschka Boedecker and Tonio Ball},
}

```
