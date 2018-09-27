# Skinapp - Skin Lesion Detection Flask App Demo

#### Dataset:
HAM10000 from ISIC archive

/ | Benign | Malignant
------------- | ------------- | -------------
original | 5365+1341 | 891+222
balanced | 891+222 | 891+222

#### Arch:
ResNext50 (pretrained)
Fine-tuned using Fastai library with data augmentation.

/ | Benign | Malignant
------------- | ------------- | -------------
original | 5365+1341 | 891+222
balanced | 891+222 | 891+222

#### Training Stat:

/ | Accuracy | Precision | ROC AUC | F1
-----------| --------- | ------- | -- | -------------
original | 92.6% | 0.825 | 0.955 | 0.727
**balanced** | 85.8% | 0.933 | 0.932 | 0.849

#### Confusion Matrix:

/ | benign | malignant
----------| --------- | -------
benign | 196 | 26
malignant | 39 | 183

## Quick-start - run it locally
1. Clone this repo, `cd` to the directory.
2. Create virtual environment with conda `conda env create`.
3. Open the terminal, run `python run.py`
4. View at http://localhost:5000.

## Acknowledge
* https://github.com/daveluo/cocoapp
* http://forums.fast.ai/t/exposing-dl-models-as-apis-microservices/13477/9
* https://github.com/fastai/fastai
