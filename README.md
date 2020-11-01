# GaitGL


# NOTE
This repo is based on [GaitSet](https://github.com/AbnerHqC/GaitSet)

## Prerequisites

- Python 3.7
- PyTorch 1.1
- CUDA 10.2

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
- **Step1:** Organize the directory as: 
`your_dataset_path/subject_ids/walking_conditions/views`.
E.g. `OUMVLP/00001/00/000/`.
- **Step2:** Cut and align the raw silhouettes with `pretreatment.py`.
the silhouettes after pretreatment **MUST have a size of 64x64**.

