import struct
import numpy as np
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn import svm
import warnings

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
    

def svm_training():
    digits = datasets.load_digits()
    classifier = svm.SVC(gamma = 0.001, C = 100)
    data = trainingList[:-10]
    labels = trainingLabels[:-10]
    
    classifier.fit(data, labels)
    
    correctAns = 0
    wrongAns = 0
    total = 0
    warnings.filterwarnings("ignore")
    
    for i in range(len(testingList)):
        correctDigit = testingLabels[i]
        prediction = classifier.predict(testingList[i])
        
        if prediction == correctDigit :
            correctAns += 1
        else:
            wrongAns += 1
            
        total += 1
        print("Test#", i, " Classifier Prediction:", prediction, ".... Correct Labels:", correctDigit)
        
    accuracy = correctAns * 100 / total
    fault = wrongAns * 100 / total
    
    
    print("Correct Accuracy:", accuracy, "%")
    print("Fault Accuracy:", fault, "%")

readData("data/mnist-train")
readLabels("data/mnist-train-labels")
svm_training()
    