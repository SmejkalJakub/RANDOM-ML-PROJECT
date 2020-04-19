import os
import glob
import sys

import numpy as np
from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from skimage.feature import hog
from skimage.io import imread


classifier = SVC(C=1e2, probability=True)


trainData = {'classes': [],
            'img_hog': [],
            'img_hog_d': []}
testData = {'classes': [],
            'img_hog': [],
            'img_hog_d': []}
evalData = {'classes': [],
            'img_hog': [],
            'img_hog_d': []}

images = {}



def loadTrainDirImages(dir, trainDir):
    os.chdir(dir)
    
    subdirectories = os.listdir(dir)

    for subdir in subdirectories:

        os.chdir(dir + '/' + subdir)

        completePath = dir + '/' + subdir

        for file in glob.glob("*.png"):
            filePath = completePath + '/' + file
            print(filePath)
            images[file] = imread(filePath, as_gray=True)
            if(trainDir):

                trainData['classes'].append(subdir)

                fd, img = hog(images[file],
                    orientations=8,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1),
                    block_norm='L1',
                    visualize=True)
                trainData['img_hog'].append(img)
                trainData['img_hog_d'].append(fd)
            else:
                testData['classes'].append(subdir)

                fd, img = hog(images[file],
                    orientations=8,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1),
                    block_norm='L1',
                    visualize=True)
                testData['img_hog'].append(img)
                testData['img_hog_d'].append(fd)

def loadEvalDirImages(dir):
    
    os.chdir(dir)

    for file in glob.glob("*.png"):
        filePath = dir + '/' + file
        print(filePath)

        images[file] = imread(filePath, as_gray=True)
        evalData['classes'] = None

        fd, img = hog(images[file],
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            block_norm='L1',
            visualize=True)
        evalData['img_hog'].append(img)
        evalData['img_hog_d'].append(fd)
    

def trainModel():
    X_trainArr = trainData['img_hog_d']
    y_trainArr = trainData['classes']

    X_testDataArr = testData['img_hog_d']
    y_testDataArr = testData['classes']

    start_time = time()
    classifier.fit(X_trainArr, y_trainArr)
    print('\n\tTraining time:', time() - start_time)

    print('\tScore:', classifier.score(X_testDataArr, y_testDataArr))


if __name__ == "__main__":

    print(sys.argv)

    training = False

    if(len(sys.argv) == 2):
        if(sys.argv[1] == '--train'):
            training = True

    currDirectory = os.getcwd()

    trainDirectory = currDirectory + '/trainImages'
    devDirectory = currDirectory + '/devImages'
    evalDirectory = currDirectory + '/evalImages'

    if(training):
        loadTrainDirImages(trainDirectory, True)
        loadTrainDirImages(devDirectory, False)
        trainModel()
    else:
        loadEvalDirImages(evalDirectory)

    X_evalDataArr = evalData['img_hog_d']
    Y_evalDataArr = evalData['classes'] 

    