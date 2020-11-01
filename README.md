# GaitGL


# NOTE
This repo is based on [GaitSet](https://github.com/AbnerHqC/GaitSet)


## Prerequisites

- Python 3.7
- PyTorch 1.1
- CUDA 10.2

### Dataset & Preparation
Download [OU-MVLP Dataset](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html).

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
- **Step1:** Organize the directory as: 
`your_dataset_path/subject_ids/walking_conditions/views`.
E.g. `OUMVLP/00001/00/000/`.
- **Step2:** Cut and align the raw silhouettes with `pretreatment_oumvlp.py`.
the silhouettes after pretreatment **MUST have a size of 64x64**.

#### Pretreatment
`pretreatment_oumvlp.py` uses the alignment method in
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).
Pretreatment your dataset by
```
python pretreatment_oumvlp.py --input_path='root_path_of_raw_dataset' --output_path='root_path_for_output'
```
- `--input_path` **(NECESSARY)** Root path of raw dataset.
- `--output_path` **(NECESSARY)** Root path for output.
- `--log_file` Log file path. #Default: './pretreatment.log'
- `--log` If set as True, all logs will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1

### Train 
Train a model by
```bash
python train.py
```
'batch_size': (32, 8), 'frame_num': 30, 'total_iter': 250000.The
learning rate is 1e − 4 in the first 150K iterations, and then is
changed into 1e − 5 for the rest of 100K iterations.
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE

### Evaluation
Evaluate the trained model by
```bash
python test_oumvlp.py
```
- `--iter` iteration of the checkpoint to load. #Default: 250000
- `--batch_size` batch size of the parallel test. #Default: 1
- `--cache` if set as TRUE all the test data will be loaded at once before the transforming start.
This might accelerate the testing. #Default: FALSE

### CAISA-E

### Dataset & Preparation
Function generate_test_gallery() generate_train_gallery() generate_test_probe() from pt_casiae.py

### Train
OUMVLP Pre-training parameters need to be added. [OUMVLP-pretrained](https://pan.baidu.com/s/1pH53yj4mfBtzmY0qPcV2uQ) key:121g  .
Train a model by
```bash
python train.py
```
'batch_size': (12, 8), 'frame_num': 64, 'total_iter': 15000. The
learning rate is 1e − 4 in the first 10K iterations, and then is
changed into 1e − 5 for the rest of 5K iterations.

### Test
Training parameters. [CASIA-E](https://pan.baidu.com/s/1DZe5yG__BS9f5PkH4jLq5w ) key:17g8 
Test a model by using Function testout() from pt_casiae.py
```bash
python pt_casiae.py
```


