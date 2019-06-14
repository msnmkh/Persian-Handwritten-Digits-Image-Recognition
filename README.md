## Introduction
This project use feature extractor and classification algorithm to detect 0 - 11 digit of  Hada dataset.
## Dataset
* HODA Farsi Digit Dataset : [http://farsiocr.ir](http://farsiocr.ir)
## Classification Algorithm
* KNN
* Bayes
* MLP
* RBF
* AdaBoost
* QDA
* GSD
* Parzen
* LinearSVC
## Feature Extrector Algorithm
* HOG
* SVD
* PCA
## Architecture
<p align="center"><img width=40% src="https://github.com/msnmkh/Persian-Handwritten-Digits-Image-Recognition/media/core-stage-of-ocr.JPG"></p>
## Code Requirements
This code is written in python. To use it you will need:
* python3
* matplotlib
* sklearn
* numpy
* cv2
## Usage
Run python PHDRI.py
## Result
Algorithm | Feature-Extrector | Accuracy |
--- | --- | ---
KNN | HOG | 91%
KNN | SVD | 94%
KNN | PCA | 85%
MLP | HOG | 93%
MLP | SVD | 57%
MLP | PCA | 90%
QDA | HOG | 31%
QDA | SVD | 20%
QDA | PCA | 16%
Parzen | HOG | 91%
Parzen | SVD | 70%
Parzen | PCA | 83.2%
Bayes | HOG | 91%
Bayes | SVD | 75%
Bayes | PCA | 85%
AdaBoost | HOG | 48%
AdaBoost | SVD | 50%
AdaBoost | PCA | 30%
LinearSVC | HOG | 85%
LinearSVC | SVD | 70%
LinearSVC | PCA | 60%
RBF | HOG | 88%
RBF | SVD | 83%
RBF | PCA | 80%


