
import struct
import random
import math
import operator
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

# Percetage of data to be used by the algorithm
TRAINING_PERCENTAGE = 60
VALIDATION_PERCENTAGE = 20
TESTING_PERCENTAGE = 20

# Global List of digits
trainingList = []
validationList = []
testingList = []

#Global List of Labels
trainingLabels = []
validationLabels = []
testingLabels = []

# Normalize features to a list
def normalizeFeatures(feature, rows, columns):
    feature = list(map(lambda x: 0 if x < 10 else 1, feature))   # Normalize
    return[feature[i:i + (rows*columns)] for i in range(0, len(feature), (rows*columns))]

# Reads MNIST File and convert it to a List
def readData(fileName):
    global trainingList, validationList, testingList
    img_file = open(fileName, 'r+b')
    # Go to beginning of file
    img_file.seek(0)
    # Get magic number
    magic_number = img_file.read(4)
    magic_number = struct.unpack('>i', magic_number)[0]

    # Get number of images
    numImages = img_file.read(4)
    # For using the 60,000 digits
    #numImages = struct.unpack('>i',numImages)[0]
    # For an arbritary amount of data
    numImages = 10000

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
    rows = struct.unpack('>i', rows)[0]

    # Get number of columns
    columns = img_file.read(4)
    columns = struct.unpack('>i', columns)[0]

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

    #close file
    img_file.close()

def readLabels(fileName):
    global trainingLabels, validationLabels, testingLabels
    label_file = open(fileName, 'r+b')
    # Go to beginning of file
    label_file.seek(0)
    # Get magic number
    magic_number = label_file.read(4)
    magic_number = struct.unpack('>i', magic_number)[0]

    # Get number of labels
    numLabels = label_file.read(4)
    numLabels = struct.unpack('>i', numLabels)[0]

    # Get Training labels
    trainingLabels = list(label_file.read(len(trainingList)))
    # Get validation labels
    validationLabels = list(label_file.read(len(validationList)))
    # Get testing labels   
    testingLabels = list(label_file.read(len(testingList)))

    label_file.close()

#displays a digit given a 1D List of 784 pixels
def displayDigit(digit):
    image = np.ndarray(shape=(28, 28))
    for k in range(28):
        for b in range(28):
            image[k, b] = digit[(k*28)+b]

    img_plot = plt.imshow(image, 'Greys')
    plt.show()

# Shows progress of a certain operation in percentage
def update_progress(progress, total):
    print('\r[{0}] {1}%'.format('#'*int((progress*30)/total), math.ceil(progress*100/total)), end="", flush=True)

# Calculates the new centroid to a cluster
def updateMeans(means, clusters):
    maxDist = -1
    for i, cluster in enumerate(clusters):
        # create empty centroid
        newCentroid = [0.0]*784
        # Calculate center point by adding everything and diving by the number of digits
        for (label, digit) in cluster:
            newCentroid = list(np.array(newCentroid) + np.array(digit))
        newCentroid = list(np.array(newCentroid) * (1.0 /(len(cluster) if len(cluster) > 0 else 1.0)))
        # calculate the distance from the old to the new centroid
        dist = 0.0
        if len(newCentroid) != 0:
            dist = distance(means[i], newCentroid)
        if dist > maxDist:
            maxDist = dist
        means[i] = newCentroid
    # Return biggest distance moved
    return maxDist

# Calculates the distance between 2 point in any dimmession
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Get the cluster index with the smallest distance to a given digit
def getBestDistanceCluster(means, digit):
    minDist = 1000000.0
    bestIndex = -1
    for i, mean in enumerate(means):
        dist = distance(digit, mean)
        if minDist > dist:
            minDist = dist
            bestIndex = i
    return bestIndex

# Classify digits according to clusters
def testKMeans(means, clusters):
    correct = 0
    wrong = 0
    clusterChosen = getBestDistanceCluster(means, testingList[0])
    for i, test in enumerate(testingList):
        clusterChosen = getBestDistanceCluster(means, test)
        if clusters[clusterChosen][0] == testingLabels[i]:
            correct += 1
        else:
            wrong += 1
    print("Correct test", (correct/len(testingList))*100,"%")
    print("wrong Tests", (wrong/len(testingList))*100, "%")

# Assigns a Label to a cluster using the most seen label
def labelCluster(cluster):
    N_DIGITS = 10
    n = [0 for c in range(N_DIGITS)]
    for (label, digit) in cluster:
        n[label] += 1
    return n.index(max(n))

# K means clustering for digit recognition
def k_means(k, trainingList):
    print("Starting K-Means Clustering with "+str(k)+ " clusters")
    # randomly select k centroid
    means = []
    for i in range(k):
        means.append(random.sample(trainingList, 1)[0][1])
    rep = False
    change = True
    meanDistChange = 10000.0
    minMeanChange = 100.0
    clusters = []
    while True:
        # if means need to be recalculated
        if rep: meanDistChange = updateMeans(means, clusters)
        # if the max distance moved was smaller than the threshold, stop clustering
        if meanDistChange <= minMeanChange:
            break
        # clean clusters
        clusters = [[] for c in range(k)]
        # For every training digit
        for i, (label, feat) in enumerate(trainingList):
            # Choose best cluster to go
            clusterChosen = getBestDistanceCluster(means, feat)
            # Assign to that cluster
            clusters[clusterChosen].append((label, feat))
            update_progress(i, len(trainingList))
        rep = True
        print(" -> max distance moved = "+ str(meanDistChange)+ ", threshold = "+str(minMeanChange))
    # Assign Labels to clusters
    for i, cluster in enumerate(clusters):
        clusters[i] = (labelCluster(cluster), cluster)
    # Show details for each cluster
    showClusteringDetails(clusters)
    # Test against test data
    testKMeans(means, clusters)

# Shows digits in each cluster class and the label trained 
def showClusteringDetails(clusters):
    print("| Ck [Label=n]| ", list(range(10)), "|")
    print("==================================================")
    for i, (label, cluster) in enumerate(clusters):
        print("| C"+str(i)+" [Label="+str(label)+"]| ", end="")
        k = [0 for c in range(10)]
        for (label2, digit) in cluster:
            k[label2] += 1
        print(k, " |")
    print("============================================")

#returns a label dataset
def labelDataset(datalist, datalabels):
    return list(zip(datalabels, datalist))

def somWinningNeuron(neurons, feat):
    minDist = 100000000.
    minIndex = -1
    for i, neuron in enumerate(neurons):
        dist = distance(neuron, feat)
        if dist < minDist:
            minDist = dist
            minIndex = i
    return i

def updateWeights(neuron, feat, a):
    for i in range(len(feat)):
        neuron[i] = neuron[i]+ (a * (feat[i] - neuron[i]))

def updateLearningCurve(a):
    a = a / 2

def labelNeurons(neurons, training):
    labels = [{c: 0 for c in range(10)} for x in range(len(neurons))]
    neuronLabels = []
    for (label, feat) in training:
        winning = somWinningNeuron(neurons, feat)
        labels[winning][label] += 1
    for lab in labels:
        neuronLabels.append(max(lab, key=lambda i: lab[i]))
    print(neuronLabels)
    return neuronLabels

def classifyDigitsSOM(neurons, neuronLabels, testing):
    correct = 0
    wrong = 0
    for (label, feat) in testing:
        winning = somWinningNeuron(neurons, feat)
        if neuronLabels[winning] == label:
            correct += 1
        else:
            wrong += 1
    print("Correct test", (correct/len(testingList))*100,"%")
    print("wrong Tests", (wrong/len(testingList))*100, "%")


# Self-Organizing Maps implementation
def somClustering(n, training, validation, testing):
    a = 1
    r = 0
    print("Starting SOM Clustering with "+str(n)+ " neurons")
    # randomly select neuron weights
    neurons = []
    for i in range(n*n):
        neurons.append(random.sample(training, 1)[0][1])

    for (label, feat) in training:
        winning = somWinningNeuron(neurons, feat)
        #update weights
        updateWeights(neurons[winning], feat, a)
        updateLearningCurve(a)


    neuronLabels = labelNeurons(neurons, training)
    classifyDigitsSOM(neurons, neuronLabels, testing)


if __name__ == '__main__':
    readData('data/mnist-train')
    readLabels('data/mnist-train-labels')
    #print(trainingList[0])
    labeledTraining = labelDataset(trainingList, trainingLabels)
    labeledTesting = labelDataset(testingList, testingLabels)
    #k_means(30, labeledTraining)
    somClustering(4, labeledTraining, validationList, labeledTesting)
