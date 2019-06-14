import cv2
from skimage.feature import hog
from sklearn.feature_selection  import SelectKBest
from sklearn.decomposition import PCA ,TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import AdaBoostClassifier

class FeatureExtraction:

    def choose(self,function,resizeImage,Labels):
        if function == "HOG":
            return self.hog(resizeImage)
        elif function == "SVD":
            return self.svd(resizeImage,Labels)
        elif function == "PCA":
            return self.pca(resizeImage,Labels)
        else:
            return self.hog(resizeImage)

    def hog(self, resizeImage):
        fd, hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
        #print(hog_image)
        im_bw = list(hog_image.flatten())
        return im_bw

    def svd(self,resizeImage,Labels):
        #unioin = FeatureUnion([("pca",),()])
        union =TruncatedSVD(n_components=2)
        fit = union.fit(resizeImage)
        fit = union.transform(resizeImage)
        im_bw=[]
        for field, possible_values in fit:
            im_bw.append(field)

        return im_bw

    def pca(self, resizeImage,Labels):
        pca = PCA(n_components=5)
        pca.fit(resizeImage)
        X = pca.transform(resizeImage)
        flatten_feature = list(X.flatten())

        return flatten_feature
