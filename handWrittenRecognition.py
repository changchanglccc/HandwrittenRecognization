import struct
import os

TRAINING_PERCENTAGE = 60
VALIDATION_PERCENTAGE = 20
TESTING_PERCENTAGE = 20

trainingList = []
validationList = []
testingList = []

def readData(fileName):
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
    numTraining = round((numImages * (TRAINING_PERCENTAGE / 100.0)), 0)
    numValidation = round((numImages * (VALIDATION_PERCENTAGE / 100.0)), 0)
    numTesting = round((numImages * (TESTING_PERCENTAGE / 100.0)), 0)

    print(str(numTraining), str(numValidation), str(numTesting), str(numTraining+numValidation+numTesting))

    # Get number of rows
    rows = img_file.read(4)
    rows = struct.unpack('>i',rows)[0]

    # Get number of columns
    columns = img_file.read(4)
    columns = struct.unpack('>i',columns)[0]

    # Get each image
    for i in range(numImages) :
        img = img_file.read(rows * columns)
        img = list(img)
        if(i < numTraining):    # Put into training set
            trainingList.append(img)
        elif i < numTraining + numValidation:   # Put into validation set
            validationList.append(img)
        else:                   # Put into testing set
            testingList.append(img)

    img_file.close()

if __name__ == '__main__':
	readData('data/mnist-data')