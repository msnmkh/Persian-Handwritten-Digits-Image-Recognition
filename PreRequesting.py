class pre:
    def SelectClassfierAndFeatureExtractor(self):
        print("Please select number of classifier:\n")  # Quadratic Discriminant Analysis
        print("1.KNN    2.Bayes     3.MLP    4.RBF   5.AdaBoost  6.QDA   7.GSD   8.Parzen  9.LinearSVC\n")
        classifier = int(input())
        if classifier == 1:
            classifier = "KNN"
        elif classifier == 2:
            classifier = "Bayes"
        elif classifier == 3:
            classifier = "MLP"
        elif classifier == 4:
            classifier = "RBF"
        elif classifier == 5:
            classifier = "AdaBoost"
        elif classifier == 6:
            classifier = "QDA"
        elif classifier == 7:
            classifier = "GSD"
        elif classifier == 8:
            classifier = "Parzen"
        elif classifier == 9:
            classifier = "LinearSVC"
        else:
            classifier = "KNN"
        print("\nPlease select number of feature extractor:\n")
        print("1.HOG     2.SVD     3.PCA \n")
        featureExtractor = int(input())
        if featureExtractor == 1:
            featureExtractor = "HOG"
        elif featureExtractor == 2:
            featureExtractor = "SVD"
        elif featureExtractor == 3:
            featureExtractor = "PCA"
        else:
            featureExtractor = "HOG"

        print("##########################################################")
        print("##############  Classifier: ", classifier, "          ###############")
        print("##############  Feature Extractor: ", featureExtractor, "   ###############")
        print("##########################################################")

        return classifier, featureExtractor
