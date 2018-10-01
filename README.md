# Skinapp - Skin Lesion Detection Flask App Demo

Demo: http://bit.ly/skinapp-demo

#### Dataset:
HAM10000 from ISIC archive

/ | Benign | Malignant
------------- | ------------- | -------------
original | 5365+1341 | 891+222
balanced | 891+222 | 891+222

#### Arch:
ResNext50 (pretrained)
Fine-tuned using Fastai library with data augmentation.

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
2. Download the [model weights](https://drive.google.com/uc?export=download&id=1K5DX2BL0k2naC47J8yfQFDMGp9EOKSAc) into `$YOUR_PATH/model`.
3. Create virtual environment with conda `conda env create -f environment.yml`.
4. Open the terminal, run `python run.py`.
5. View at http://localhost:5000.

## Acknowledgement
* https://github.com/daveluo/cocoapp
* http://forums.fast.ai/t/exposing-dl-models-as-apis-microservices/13477/9
* https://github.com/fastai/fastai
