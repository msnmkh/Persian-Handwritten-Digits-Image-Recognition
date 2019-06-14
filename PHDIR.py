import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from FeatureExtraction import FeatureExtraction
from sklearn.model_selection import train_test_split
from Classifier import Classifer
from PreRequesting import pre

def ExtractFeatureAndBuildDataset(extract_feature="HOG",size=50):
    DATA = []
    Labels = []
    for path, subdirs, files in os.walk("persian_digit"):
        for name in files:
            imagePath = os.path.join(path, name)
            # Read image as gray scale
            img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
            # Resize image for get better feature
            resizeImg = cv2.resize(img,(size,size), interpolation=cv2.INTER_AREA)
            # Thredshold for 128 is 0 and 255 is
            _, IMG = cv2.threshold(resizeImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Append extract featured DATA
            DATA.append(FeatureExtraction().choose(extract_feature,IMG,Labels))
            # Label of each image
            Labels.append(path[14:])

    return DATA,Labels

def main():
    print("########################################################")
    print("---------------------- O - C - R------------------------")
    print("-------------  Persian Digit Recognition  --------------")
    print("########################################################\n")
    # Size of resizing image
    size = 50
    cnt=0
    while(1):
        # Python does not support do-while!
        if(cnt!=0):
            print("\n-------------------------------------------------- F-I-N-I-S-H-E-D -------------------------------------------------\n")
        # Select classfier and feature extractor
        classifier , feature_selector =pre().SelectClassfierAndFeatureExtractor()
        # Extract feature of image
        DATA,Labels = ExtractFeatureAndBuildDataset(feature_selector,size)
        # Set train and test data
        X_train, X_test, y_train, y_test = train_test_split(DATA, Labels, test_size=0.10)
        cls = Classifer().choose(classifier, X_train, X_test, y_train, y_test)
        cnt+=1
if __name__=="__main__":
    main()
