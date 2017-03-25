import struct
import random
import math
import operator
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

TRAINING_PERCENTAGE = 60
VALIDATION_PERCENTAGE = 20
TESTING_PERCENTAGE = 20

trainingList = []
validationList = []
testingList = []

trainingLabels = []
validationLabels = []
testingLabels = []

def normalizeFeatures(feature, rows, columns):
    feature = list(map(lambda x: 0 if x < 100 else 1, feature))   # Normalize
    return[feature[i:i + (rows*columns)] for i in range(0, len(feature), (rows*columns))]

def readData(fileName):
    global trainingList, validationList, testingList
    img_file = open(fileName,'r+b')
    # Go to beginning of file
    img_file.seek(0)
    # Get magic number
    magic_number = img_file.read(4)
    magic_number = struct.unpack('>i',magic_number)[0]

    # Get number of images
    numImages = img_file.read(4)
    numImages = 3000#struct.unpack('>i',numImages)[0]

    #calculate size of training, validation and testing sets
    numTraining = int(round((numImages * (TRAINING_PERCENTAGE / 100.0)), 0))
    numValidation = int(round((numImages * (VALIDATION_PERCENTAGE / 100.0)), 0))
    numTesting = int(round((numImages * (TESTING_PERCENTAGE / 100.0)), 0))

    print('Training Amount: ' + str(numTraining),
     '\nValidation Amount: ' +str(numValidation),
     '\nTesting Amount: ' + str(numTesting),
     '\nTotal: ' + str(numTraining+numValidation+numTesting))

    # Get number of rows
    rows = img_file.read(4)
    rows = struct.unpack('>i',rows)[0]

    # Get number of columns
    columns = img_file.read(4)
    columns = struct.unpack('>i',columns)[0]

    # Get Training portion
    print('Reading & Parsing Training data...')
    trainingList = img_file.read(rows * columns * numTraining)
    trainingList = normalizeFeatures(trainingList, rows, columns)

    # Get Validation portion
    print('Reading & Parsing Validation data...')
    validationList = img_file.read(rows * columns * numValidation)
    validationList = normalizeFeatures(validationList, rows, columns)

    # Get Testing portion
    print('Reading & Parsing Testing data...')
    testingList = img_file.read(rows * columns * numTesting)
    testingList = normalizeFeatures(testingList, rows, columns)

    # uncomment to see a given digit
    # image = np.ndarray(shape=(rows,columns))

    # for i in range(rows):
    #     for j in range(columns):
    #         image[i,j] = trainingList[1][(i*columns)+j]

    # img_plot = plt.imshow(image,'Greys')
    # plt.show()

    #close file
    img_file.close()

def update_progress(progress):
    print('\r[{0}] {1}%'.format('#'*int((progress*30)/len(trainingList)), math.ceil(progress*100/len(trainingList))), end="", flush=True)

def readLabels(fileName):
    global trainingLabels, validationLabels, testingLabels
    label_file = open(fileName,'r+b')
    # Go to beginning of file
    label_file.seek(0)
    # Get magic number
    magic_number = label_file.read(4)
    magic_number = struct.unpack('>i',magic_number)[0]

    # Get number of labels
    numLabels = label_file.read(4)
    numLabels = struct.unpack('>i',numLabels)[0]

    # Get Training labels
    trainingLabels = list(label_file.read(len(trainingList)))
    # Get validation labels
    validationLabels = list(label_file.read(len(validationList)))
    # Get testing labels   
    testingLabels = list(label_file.read(len(testingList)))

    # uncomment to see a given label
    #print(trainingLabels[6])

    label_file.close()

def updateMeans(means, clusters):
    if(len(clusters) == 0):
        return
    meanCentroid = [[0.0]*784]*10
    for point in clusters:
        meanCentroid[point[1]] = map(lambda x: reduce(sum, x) % 2, zip(point[0], meanCentroid))

def getBestDistanceCluster(means, score):
    minDist = 100000.0
    bestIndex = -1
    for i, mean in enumerate(means):
        dist = math.sqrt(reduce(lambda x, y: x + y, map(lambda val: math.pow(reduce(lambda x, y: x-y, val), 2),zip(score, mean[1]))))
        if(minDist > dist):
            minDist = dist
            bestIndex = i
    return bestIndex

def k_means():
    # randomly select mean for each digit class
    means = []
    for i in range(10):
        digitsForClass = list(filter(lambda x: x[0] == i, trainingList))
        means.append(list(digitsForClass[random.randint(0, len(digitsForClass)-1)]))
    clusters = []
    rep = False
    change = True
    o = 0
    while change:
        o += 1
        print(o)
        change = False
        updateMeans(means, clusters)
        for i, feat in enumerate(trainingList):
            clusterChosen = getBestDistanceCluster(means, feat[1])
            if not rep:
                clusters.append([feat[1], clusterChosen])
                change = True
            elif clusters[i][1] != clusterChosen:
                clusters[i][1] = clusterChosen
                change = True
            update_progress(i)
        rep = True
    # plot graph
    # uncomment to see a given digit
    image = np.ndarray(shape=(rows,columns))

    for i in range(28):
        for j in range(28):
            image[i,j] = means[0][(i*columns)+j]

    img_plot = plt.imshow(image,'Greys')
    plt.show()
    

def labelAndSort():
    global trainingList
    trainingList = sorted(zip(trainingLabels, trainingList), key=(lambda x: x[0]))


if __name__ == '__main__':
    readData('data/mnist-train')
    readLabels('data/mnist-train-labels')
    # Label and order
    labelAndSort()
    k_means()