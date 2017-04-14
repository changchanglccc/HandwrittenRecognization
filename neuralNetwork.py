
# Neural Network by using pybrain
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import *
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter

import handWrittenRecognition
import MNIST_Data
# create an Object to get the data source
dataObject = MNIST_Data.MNIST_Processing()
traininglist = dataObject.neural_data_set
traininglabels = dataObject.neural_label_set

# step1
#create neural network
fnn = FeedForwardNetwork()

#set three layers, input+ hidden layer+ output  28*28=784

# the first feature extraction
#inLayer = LinearLayer(784,name='inLayer')
# the second feature extraction
inLayer = LinearLayer(28,name='inLayer')
hiddenLayer = SigmoidLayer(30,name = 'hiddenLayer0')
outLayer = LinearLayer(10, name = 'outLayer')

#There are a couple of different classes of layers. For a complete list check out the modules package.

#add these three Layers into neural network
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
fnn.addOutputModule(outLayer)

#create the connections between three layers
in_to_hidden = FullConnection(inLayer,hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer,outLayer)

#add connections into network
fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_out)

#make neural network available
fnn.sortModules()
'''
print(fnn)
print()
print(fnn.activate([1,2]))
print(in_to_hidden.params)
print(fnn.params)
print("=================================================")
'''
#step2： construct data set

# define that the input of data set is 784 demensions, output is 10 demension
# the first feature extraction
#DS = SupervisedDataSet(784,10)
# the first feature extraction
DS = SupervisedDataSet(28,10)

#add sample data set
for i in range(len(traininglist)):
    DS.addSample(traininglist[i], traininglabels[i])


X = DS['input']
Y = DS['target']
dataTrain, dataTest = DS.splitWithProportion(0.8)
xTrain,yTrain = dataTrain['input'],dataTrain['target']
xTest,yTest = dataTest['input'],dataTest['target']

#step3
# trainner use BP algorithm
verbose = True
trainer = BackpropTrainer(fnn, dataTrain, verbose = True, learningrate = 0.5,lrdecay= 0.5, momentum=0)
trainer.trainUntilConvergence(DS,maxEpochs=10)



NetworkWriter.writeToFile(fnn,'networkClassifier.txt')

print("#############")
out = fnn.activateOnDataset(DS)
print(out)
# u can give an input, and use activate function to see the result.
#fnn.activate(input)


# Reference: http://pybrain.org/docs/tutorial/netmodcon.html#examining-a-network; https://www.zengmingxia.com/use-pybrain-to-fit-neural-networks/