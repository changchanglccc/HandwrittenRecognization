import struct
import os

TRAINING_PERCENTAGE = 60
VALIDATION_PERCENTAGE = 20
TESTING_PERCENTAGE = 20

trainingList = []
validationList = []
testingList = []

def normalizeFeatures(feature):
    return 0 if feature < 50 else 1

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

    print(str(numTraining), str(numValidation), str(numTesting), str(numTraining+numValidation+numTesting))

    # Get number of rows
    rows = img_file.read(4)
    rows = struct.unpack('>i',rows)[0]

    # Get number of columns
    columns = img_file.read(4)
    columns = struct.unpack('>i',columns)[0]

    # Get Training portion
    trainingList = img_file.read(rows * columns * numTraining)
    trainingList = list(map(normalizeFeatures, trainingList))   # Normalize

    trainingList = [trainingList[i:i + (rows*columns)] for i in range(0, len(trainingList), (rows*columns))]

    # Get Validation portion
    validationList = img_file.read(rows * columns * numValidation)
    validationList = list(map(normalizeFeatures, validationList))   # Normalize

    validationList = [validationList[i:i + (rows*columns)] for i in range(0, len(validationList), (rows*columns))]

    # Get Testing portion
    testingList = img_file.read(rows * columns * numTesting)
    testingList = list(map(normalizeFeatures, testingList))     # Normalize

    testingList = [testingList[i:i + (rows*columns)] for i in range(0, len(testingList), (rows*columns))]

    #close file
    img_file.close()

if __name__ == '__main__':
	readData('data/mnist-train')