import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

jagged_list = [['a','b','c','d','e'],
               [1,2,3],
               ['apple','berry','cow','dog']]

def truncate_2d_to_shortest_item(list):
    newlist = []
    if len(list) >= 2:
        min_len = len(list[0])
        for l in range(1,len(list)):
            min_len = min(min_len,len(list[l]))
        for sub_list in range(0,len(list)):
            if len(list[sub_list]):
                newlist.append(list[sub_list][:min_len])
                #del list[sub_list][min_len:]
    else:
        print("list is already truncated")
    return newlist

print('jagged list: ',jagged_list)
print('truncated jagged list: ',truncate_2d_to_shortest_item(jagged_list))

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
        if os.path.isdir(os.path.join(classes_dir,class_dir)):
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
                #plt.show()
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

import time
time.sleep(5)
feature_vectors, labels = get_training_data('/Users/ryoungblood/cio-images/cimg/tests/data/images/classifier_images/')

print(len(feature_vectors))

truncated_feature_vectors = truncate_2d_to_shortest_item(feature_vectors)
# truncated_feature_vectors = [[1,2,3],
#                              [4,5,6],
#                              [7,8,9]]
# labels = [0,0,1]

time.sleep(1)
print('num images: ',len(truncated_feature_vectors))
time.sleep(1)
print('num feature vectors in first image: ',len(truncated_feature_vectors[0]))
time.sleep(1)
print('num feature vectors in second image: ',len(truncated_feature_vectors[1]))
time.sleep(1)
print('num labels: ',len(labels))
time.sleep(1)
print('num features in first vector of first image: ',len(truncated_feature_vectors[0][0]))
print('first feature vector of fisr image: ',truncated_feature_vectors[0][0])

# print('='*50)
# # print('feature vectors: ',feature_vectors)
# # print('-'*50)
# # print('labels: ',labels)
#
svm_classifier = svm.LinearSVC().fit(truncated_feature_vectors,labels)

# so basically, this is erroring out because I am giving a 3D feature set, and SVM
# fit wants at most 2D (a table) for the training features.  The reason mine is 3D
# is that each row in the table represents an image, and each image has multiple keypoints,
# and each keypoint is represented by 128 ints.  Now certainly I could flatten that down
# to 2D, but it would be arbitrary because the keypoints within each image are not
# ordered (just as my selection of the first n of them from each image was arbitrary).
# The ints in each keypoint descriptor are ordered, but even if I merged those into a single
# value without any information loss, then I would have 2D training data where each column
# in my table would be a keypoint, but the column ordering between rows is meaningless and
# arbitrary.  Therefore, I need a better way.  Off to learn more about BOW.