import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

def get_training_data(classes_dir,detector_alg='sift',descriptor_alg='sift'):
    if detector_alg == 'sift':
        detector_alg = cv2.xfeatures2d.SIFT_create()
    elif detector_alg == 'surf':
        detector_alg = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("invalid detector_alg")

    feature_vectors = []
    labels = []

    for class_num, class_dir in enumerate(os.listdir(classes_dir)):
        for image_name in os.listdir(os.path.join(classes_dir,class_dir)):
            labels.append(class_num)

            image_filepath = os.path.join(classes_dir,class_dir,image_name)
            image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)

            keypoints = detector_alg.detect(image)
            keypoints, descriptors = detector_alg.compute(image,keypoints)
            feature_vectors.append(descriptors)

            print(image_name)
            print(os.path.split(class_dir)[1])
            print(image_filepath)
            print('num keypoints: ',len(keypoints))
            print('num descriptors: ',len(descriptors))

            plt.imshow(cv2.drawKeypoints(image,keypoints,
                                         (255,0,0)))
            plt.show()
    return feature_vectors,labels

def train_model():
    """
    STEPS:
    #1 create model object
    #2 train model object based on results of get_training_data
    #3 get sliding window (scale & space) set of negative images
    #4 apply classifier to set from #3
    #5 take any false positive results put their desriptors in the negative set
        #Alternatively, put the their images in the negative folder and retrain..
    #6 apply new classifier to test set
    """


feature_vectors, labels = get_training_data('/Users/ryoungblood/cio-images/cimg/tests/data/images/classifier_images/')

print('='*50)
# print('feature vectors: ',feature_vectors)
# print('-'*50)
# print('labels: ',labels)

svm_classifier = svm.LinearSVC().fit(feature_vectors,labels)


