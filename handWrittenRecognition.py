import struct
import os

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
    list(map(lambda x: 0 if x < 50 else 1, feature))   # Normalize
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
    numImages = struct.unpack('>i',numImages)[0]

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
    normalizeFeatures(trainingList, rows, columns)

    # Get Validation portion
    print('Reading & Parsing Validation data...')
    validationList = img_file.read(rows * columns * numValidation)
    normalizeFeatures(validationList, rows, columns)

    # Get Testing portion
    print('Reading & Parsing Testing data...')
    testingList = img_file.read(rows * columns * numTesting)
    normalizeFeatures(testingList, rows, columns)

    #close file
    img_file.close()

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

    label_file.close()


if __name__ == '__main__':
    readData('data/mnist-train')
    readLabels('data/mnist-train-labels')