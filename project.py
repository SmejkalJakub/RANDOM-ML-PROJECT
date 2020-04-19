import os
import glob

import numpy as np
from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from skimage.feature import hog
from skimage.io import imread


classifier = SVC(C=1e2, probability=True)


trainData = {'targets': [],
            'img_hog': [],
            'img_hog_d': []}
testData = {'targets': [],
            'img_hog': [],
            'img_hog_d': []}

images = {}



def loadTrainDirData(dir, trainDir):
    os.chdir(dir)
    for file in glob.glob("*.png"):
        filePath = dir + '/' + file
        images[file] = imread(filePath, as_gray=True)
        if(trainDir):

            trainData['targets'].append(file[:4:])

            fd, img = hog(images[file],
                   orientations=8,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(1, 1),
                   block_norm='L1',
                   visualize=True)
            trainData['img_hog'].append(img)
            trainData['img_hog_d'].append(fd)
        else:
            testData['targets'].append(file[:4:])

            fd, img = hog(images[file],
                   orientations=8,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(1, 1),
                   block_norm='L1',
                   visualize=True)
            testData['img_hog'].append(img)
            testData['img_hog_d'].append(fd)

if __name__ == "__main__":

    currDirectory = os.getcwd()
    trainDirectory = currDirectory + '/trainImages'
    devDirectory = currDirectory + '/devImages'

    loadTrainDirData(trainDirectory, True)
    loadTrainDirData(devDirectory, False)

    X_trainArr = trainData['img_hog_d']
    y_trainArr = trainData['targets']
    
    X_testData = testData['img_hog_d']
    y_testData = testData['targets']

    start_time = time()
    classifier.fit(X_trainArr, y_trainArr)
    print('\n\tTraining time:', time() - start_time)

    print('\tScore:', classifier.score(X_test, y_test))
