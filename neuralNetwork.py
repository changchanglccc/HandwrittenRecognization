# Neural Network by using pybrain
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import *
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.supervised.trainers import BackpropTrainer

# step1
#create neural network
fnn = FeedForwardNetwork()

#set three layers, input+ hidden layer+ output
inLayer = LinearLayer(2,name='inLayer')
hiddenLayer = SigmoidLayer(3,name = 'hiddenLayer0')
outLayer = LinearLayer(1, name = 'outLayer')

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
print(fnn)
print()
print(fnn.activate([1,2]))
print(in_to_hidden.params)
print(fnn.params)
print("=================================================")

#tutorial using Recurrent Networks
fnn = RecurrentNetwork()
fnn.addInputModule(LinearLayer(2,name = 'in'))
fnn.addModule(SigmoidLayer(3,name = 'hidden'))
fnn.addOutputModule(LinearLayer(1, name = 'out'))
fnn.addConnection(FullConnection(fnn['in'],fnn['hidden'], name = 'c1'))
fnn.addConnection(FullConnection(fnn['hidden'],fnn['out'], name = 'c2'))
fnn.addRecurrentConnection(FullConnection(fnn['hidden'],fnn['hidden'], name = 'c3'))
fnn.sortModules()
print(fnn.activate((2,2)))
print(fnn.activate((2,2)))
print(fnn.activate((2,2)))

'''
#step2： construct data set
# define that the input of data set is 3 demensions, output is one demension
DS = SupervisedDataSet(3,1)
#add sample data set
for i in len(y):
    DS.addSample([x1[i],x2[i],x3[i]],[y[i]])

dataTrain, dataTest = DS.splitWithProportion(0.8)
xTrain,yTrain = dataTrain['input'],dataTrain['target']
xTest,yTest = dataTest['input'],dataTest['target']

#step3
# trainner use BP algorithm
verbose = True
trainer = BackpropTrainer(fnn, dataTrain, verbose = True, learningrate = 0.01)
# maxEpochs : 1000
trainer.trainUnyilConvergence(maxExpochs = 10000)
'''

# Reference: http://pybrain.org/docs/tutorial/netmodcon.html#examining-a-network