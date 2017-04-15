import math
import struct
import numpy
import matplotlib.pyplot
import random
import sys

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

#For GeneticAlgorithm
fitnessScore, couple, children, population, populationLabel = [], [], [], [], []

# Normalize features to a list
def normalizeFeatures(feature, rows, columns, option):
    feature = list(map(lambda x: 0 if x < 10 else 1, feature))   # Normalize

    data = [feature[i:i + (rows*columns)] for i in range(0, len(feature), (rows*columns))]

    if(option == 'pixelsPerRow'):
        data = pixelsPerRowFeatExtraction(data)

    return data

def pixelsPerRowFeatExtraction(data):
    totalData =[]
    rowData = []
    row = 0
    for d in data:
        rowData = []
        for i in range(28):
            row = 0
            for j in range(28):
                if(d[i*28 + j] == 1):
                    row += 1
            rowData.append(row)
        totalData.append(rowData)
    return totalData

# Reads MNIST File and convert it to a List
def readData(fileName, featExt):
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
    trainingList = normalizeFeatures(trainingList, rows, columns, featExt)

    # Get Validation portion
    print('Reading & Parsing Validation data...')
    validationList = img_file.read(rows * columns * numValidation)
    validationList = normalizeFeatures(validationList, rows, columns, featExt)

    # Get Testing portion
    print('Reading & Parsing Testing data...')
    testingList = img_file.read(rows * columns * numTesting)
    testingList = normalizeFeatures(testingList, rows, columns, featExt)

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
    image = numpy.ndarray(shape=(28, 28))
    for k in range(28):
        for b in range(28):
            image[k, b] = digit[(k*28)+b]
    img_plot = matplotlib.pyplot.imshow(image, 'Greys')
    matplotlib.pyplot.show()

##Fitness Function is calculated by the average of 1 contained per label.
def fitnessList():
    for num in range(0,10):
        totalOnes = 0;
        count = 0;
        print("Label : " + str(num))
        for i in range(len(trainingLabels)):
            if num == trainingLabels[i]:
                count += 1
                totalOnes += trainingList[i].count(1)
        print("Added Average : " + str(totalOnes/(len(trainingList[num]) * count)))
        fitnessScore.append(totalOnes/(len(trainingList[num]) * count))

#For debug purpose
def displayList(list):
    for i in list:
        print(i)

# From a list, take two individuals as parents and create new children
def crossOver(evaList):
    x = random.choice(range(0,len(evaList)))
    y = random.choice(range(0,len(evaList)))
    couple.append(evaList[x])
    couple.append(evaList[y])
    parentX = evaList[x]
    parentY = evaList[y]
    crossPos = random.choice(range(0,len(evaList[x])))
    print('ParentX: ' + str(trainingLabels[x]) +
          '\nParentY: ' + str(trainingLabels[y]) +
          '\nCrossing Position: ' + str(crossPos))
    children.append(couple[0][57:] + couple[1][:57])
    children.append(couple[1][57:] + couple[0][:57])

#Return the fitness of the individual
def checkFitness(individual):
    return (individual.count(1)/len(individual))

#child is the index of the children list and rate in percentage
def mutate(child, rate):
    mutatedGene = len(children[child]) * rate / 100
    for mg in range(0, int(mutatedGene)):
        rand = random.choice(range(0, len(children[child])))
        if children[child][rand] == 1:
             children[child][rand] = 0
        if children[child][rand] == 0:
            children[child][rand] =1

def newPopulation(pop, parents, children):
    population = list(pop)
    childAHealth = checkFitness(children[0])
    childBHealth = checkFitness(children[1])
    parentXHealth = checkFitness(parents[0])
    parentYHealth = checkFitness(parents[1])
    indX = (population.index(parents[0]))
    indY = (population.index(parents[1]))
    if(((childAHealth > parentXHealth) & (childAHealth > parentYHealth)) & (childBHealth > parentXHealth) & (childBHealth > parentYHealth)):
        for index in range(0, len(pop)):
            if index == indX:
                population[index] = children[0]
                print("ParentX replaced with ChildA")
            if index == indY:
                population[index] = children[1]
                print("ParentY replaced with ChildB")
    if((childAHealth > parentXHealth) & (childAHealth <= parentYHealth)):
        for index in range(0, len(pop)):
            if index == indX:
                population[index] = children[0]
                print("ParentX replaced with ChildA")
    if((childBHealth > parentXHealth) & (childBHealth <= parentYHealth)):
        for index in range(0, len(pop)):
            if index == indX:
                population[index] = children[1]
                print("ParentX replaced with ChildB")
    if((childAHealth <= parentXHealth) & (childAHealth > parentYHealth)):
        for index in range(0, len(pop)):
            if index == indY:
                population[index] = children[0]
                print("ParentY replaced with ChildA")
    if((childBHealth <= parentXHealth) & (childBHealth > parentYHealth)):
        for index in range(0, len(pop)):
            if index == indY:
                population[index] = children[1]
                print("ParentY replaced with ChildB")
    if(((childAHealth < parentXHealth) & (childAHealth < parentYHealth)) & ((childBHealth < parentXHealth) & (childBHealth < parentYHealth))):
        print("Children are weaker than parents, population remains the same")
    return population

#Return the percentage of accuracy when classifing as numbers
def predict(list):
    for i in range(0,len(list)):
        num = (list[i].count(1)/len(list[i]))
        tmpScore = 5.0
        tmpIndex = 10
        for j in range(0,len(fitnessScore)):
            tmp = abs(fitnessScore[j] - num)
            if(tmp < tmpScore):
                tmpScore = tmp
                tmpIndex = j
        populationLabel.append(tmpIndex)

def accuracy(resultLabel, originalLabel):
    match = 0
    for i in range(0,len(originalLabel)):
        if(resultLabel[i] == originalLabel[i]):
            match += 1
    print((match/len(originalLabel))*100)

def emptyLists():
    couple[:] = []
    children[:] = []
    population[:] = []
    populationLabel[:] = []

if __name__ == '__main__':
    featExtArg = ''
    if(len(sys.argv) > 1):
        featExtArg = str(sys.argv[1])
    readData('data/mnist-train', featExtArg)
    readLabels('data/mnist-train-labels')
    print("Training for Fitness Score of the number")
    fitnessList()
    print("------------------------------------------")
    print("CrossOver with Training Set")
    crossOver(trainingList)
    print("Checking Children Fitness and Generate new population")
    print("Clustering new population")
    predict(newPopulation(trainingList, couple, children))
    print("Accuracy of the Clustering with Training Set")
    accuracy(populationLabel,trainingLabels)
    emptyLists()
    print("------------------------------------------")
    print("CrossOver with Validation Set")
    crossOver(validationList)
    print("Checking Children Fitness and Generate new population")
    print("Clustering new population")
    predict(newPopulation(validationList, couple, children))
    print("Accuracy of the Clustering with Training Set")
    accuracy(populationLabel,validationLabels)
    emptyLists()
    print("------------------------------------------")
    print("CrossOver with Testing Set")
    crossOver(testingList)
    print("Checking Children Fitness and Generate new population")
    print("Clustering new population")
    predict(newPopulation(testingList, couple, children))
    print("Accuracy of the Clustering with Testing Set")
    accuracy(populationLabel,testingLabels)
