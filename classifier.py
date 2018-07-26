import imutils
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import pickle
import shutil


class KNNImageClassifier:
    """
    A Classifier for clustering images using KNN Algorithm
    """

    def __init__(self):
        # default folder name
        self.imagepath_name = "train"
        # initialize the raw pixel intensities matrix, the features matrix,
        # and labels list
        self.rawImages = []
        self.features = []
        self.labels = []

        # initialization for raw pixel based training and testing while learning
        self.trainRI = None
        self.testRI = None
        self.trainRL = None
        self.testRL = None

        # initialization for feature based training and testing while learning
        self.trainFeat = None
        self.testFeat = None
        self.trainLabels = None
        self.testLabels = None

        # for learnt model object
        self.learnt_model = None

    def image_to_feature_vector(self, image, size=(32, 32)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, size).flatten()

    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])

        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)

        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
            cv2.normalize(hist, hist)

        # return the flattened histogram as the feature vector
        return hist.flatten()

    def train_test_split_data(self, rawImages, features, labels, test_size=0.25, random_state=42):
        # for rawpixel based training you need labels and rawImages
        (self.trainRI, self.testRI, self.trainRL, self.testRL) = train_test_split(rawImages, labels, test_size=0.25,
                                                                                  random_state=42)
        # for feature based training you need labels and features
        (self.trainFeat, self.testFeat, self.trainLabels, self.testLabels) = train_test_split(features, labels,
                                                                                              test_size=0.25,
                                                                                              random_state=42)

    def image_preprocessing(self, imagePaths):
        # initialize lists
        rawImages = []
        features = []
        labels = []
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label (assuming that our
            # path as the format: /path/to/dataset/{class}.{image_num}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-1].split(".")[0]

            # extract raw pixel intensity "features", followed by a color
            # histogram to characterize the color distribution of the pixels
            # in the image
            pixels = self.image_to_feature_vector(image)
            hist = self.extract_color_histogram(image)

            # update the raw images, features, and labels matricies,
            # respectively
            rawImages.append(pixels)
            features.append(hist)
            labels.append(label)

            # show an update every 1,000 images
            if i > 0 and i % 100 == 0:
                print("[INFO] processed {}/{}".format(i, len(imagePaths)))

        # show some information on the memory consumed by the raw images
        # matrix and features matrix
        self.rawImages = np.array(rawImages)
        self.features = np.array(features)
        self.labels = np.array(labels)

        print("[INFO] pixels matrix: {:.2f}MB".format(self.rawImages.nbytes / (1024 * 1000.0)))
        print("[INFO] features matrix: {:.2f}MB".format(self.features.nbytes / (1024 * 1000.0)))

        # cross validation data split for train and test
        # for raw pixels train and test data and features train and test data
        self.train_test_split_data(rawImages=self.rawImages, labels=self.labels, features=self.features)

        # calling the main two learning functionality
        # train model uisng image feature
        self.train_evaluate_for_features()
        # train model using raw pixels of the image
        self.train_evaluate_for_rawpixels_intensities()

    def train_evaluate_for_rawpixels_intensities(self):
        # train and evaluate a k-NN classifer on the raw pixel intensities
        print("[INFO] evaluating raw pixel accuracy...")
        model = KNeighborsClassifier(n_neighbors=4, n_jobs=1)
        model.fit(self.trainRI, self.trainRL)
        acc = model.score(self.testRI, self.testRL)
        print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

    def train_evaluate_for_features(self):
        # train and evaluate a k-NN classifer on the histogram
        # representations
        print("[INFO] evaluating histogram accuracy...")
        model = KNeighborsClassifier(n_neighbors=4, n_jobs=1)
        model.fit(self.trainFeat, self.trainLabels)
        acc = model.score(self.testFeat, self.testLabels)
        print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

        # setting the learnt model
        self.learnt_model = model

        # Now pickle learnt model
        self.pickle_unpickle_model(pick=True)

    def start_processing(self, imagepath_name=None):
        print("[INFO] describing images...")
        if imagepath_name:
            self.imagepath_name = imagepath_name
        imagePaths = list(paths.list_images(self.imagepath_name))
        self.image_preprocessing(imagePaths)

    def pickle_unpickle_model(self, name='knn_model.obj', pick=False, unpick=False):
        if pick:
            with open(name, 'wb') as fp:
                pickle.dump(self.learnt_model, fp)
        if unpick:
            try:
                with open(name, 'rb') as fp:
                    self.learnt_model = pickle.load(fp)
            except FileExistsError:
                pass

    def detect_data(self):
        if self.learnt_model:
            dirname = "input"
            imagePaths = list(paths.list_images(dirname))
            if os.path.isdir(dirname):
                # search output directory and delete that then recreate the folder
                output_dir = 'affected'
                if os.path.isdir(output_dir):
                    shutil.rmtree(os.path.join(os.getcwd(), output_dir))
                    os.mkdir(output_dir)
                else:
                    os.mkdir(output_dir)
                input_data_list = []
                for (i, imagePath) in enumerate(imagePaths):
                    data = dict()
                    data['file_name'] = imagePath
                    test_image = cv2.imread(imagePath)
                    cal_histogram = self.extract_color_histogram(test_image)
                    predicted_label = self.learnt_model.predict(np.array(cal_histogram).reshape(1, -1))
                    accuracy_vect = self.learnt_model.predict_proba(np.array(cal_histogram).reshape(1, -1))
                    confid = accuracy_vect[0][0] if accuracy_vect[0][0] >= accuracy_vect[0][1] else \
                        accuracy_vect[0][1]
                    if accuracy_vect[0][0] == accuracy_vect[0][1]:
                        data['predicted_label'] = 'Not in Any Class'
                        data['confidence'] = str(float(confid) * 100)
                    else:
                        if predicted_label[0] == "affected":
                            data['predicted_label'] = predicted_label[0]
                            data['confidence'] = str(float(confid) * 100)

                            # Now move the affected files to that folder
                            shutil.copy(imagePath, os.path.join(os.getcwd(), output_dir))

                        if predicted_label[0] == "healthy":
                            data['predicted_label'] = predicted_label[0]
                            data['confidence'] = str(float(confid) * 100)
                    input_data_list.append(data)
                file = open('classification_report.csv', 'w')
                for item in input_data_list:
                    line = ""
                    for ele in item:
                        line += item[ele] + ","
                    line += "\n"
                    file.write(line)
                file.close()
            else:
                print("No folder is present here named 'input' ")
        else:
            self.start_processing()

    def classify_data(self):
        if os.path.exists('knn_model.obj'):
            self.pickle_unpickle_model(unpick=True)
            if self.learnt_model:
                self.detect_data()
            else:
                self.start_processing()
                self.detect_data()
        else:
            self.start_processing()
            if self.learnt_model:
                self.detect_data()


# Execution
KNN = KNNImageClassifier()
KNN.classify_data()
