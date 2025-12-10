# Multi-BK-Net-general-EEG-pathology-classification
Code accompanies the paper [Multi-BK-Net: Multi-Branch Multi-Kernel Convolutional Neural Networks for Clinical EEG Analysis](https://openreview.net/forum?id=IsG10xZAaA).
A short video presentation of the paper is provided [here](https://www.youtube.com/watch?v=KeDA_GqvEng).


## Abstract

Classifying an electroencephalography (EEG) recording as pathological or non-pathological is an important first step in diagnosing and managing neurological diseases and disorders. As manual EEG classification is costly, time-consuming and requires highly trained experts, deep learning methods for automated classification of general EEG pathology offer a promising option to assist clinicians in screening EEGs. Convolutional neural networks (CNNs) are well-suited for classifying pathological EEG signals due to their ability to perform end-to-end learning. In practice, however, current CNN solutions suffer from limited classification performance due to I) a single-scale network design that cannot fully capture the high intra- and inter-subject variability of the EEG signal, the diversity of the data, and the heterogeneity of pathological EEG patterns and II) the small size and limited diversity of the dataset commonly used to train and evaluate the networks. These challenges result in a low sensitivity score and a performance drop on more diverse patient populations, further hindering their reliability for real-world applications.
Here, we propose a novel multi-branch, multi-scale CNN called Multi-BK-Net (Multi-Branch Multi-Kernel Network), comprising five parallel branches that incorporate temporal convolution, spatial convolution, and pooling layers, with temporal kernel sizes defined by five clinically relevant frequency bands in its first block. 
Evaluation is based on two public datasets with predefined test sets: the Temple University Hospital (TUH) Abnormal EEG Corpus and the TUH Abnormal Expansion Balanced EEG Corpus. 
Our Multi-BK-Net outperforms five baseline architectures and state-of-the-art end-to-end approaches in terms of accuracy and sensitivity on these datasets, setting a new benchmark. Furthermore, ablation experiments highlight the importance of the multi-branch, multi-scale input block of the Multi-BK-Net. Overall, our findings indicate the efficacy of multi-branch, multi-scale CNNs in accurately and reliably classifying EEG pathology within the evaluated datasets, demonstrating advantages in handling data heterogeneity compared to other deep learning approaches.  Thus, this study contributes to the ongoing development of deep end-to-end methods for general EEG pathology classification.


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
  url = {https://openreview.net/forum?id=IsG10xZAaA},
  author = {Ann-Kathrin Kiessner and Joschka Boedecker and Tonio Ball},
}

```
