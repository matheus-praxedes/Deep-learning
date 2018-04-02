import numpy as np
import os.path
from data import Instance, DataSet
from neural_network import NeuralNetwork
from activation_function import ActivationFunction


#Building data set 1

data_set_1 = DataSet("data_set_1")

if (os.path.isfile(data_set_1.name)):
	data_set_1.loadFile()

else:
	data_set_size_1 = 1000
	for i in range(0, data_set_size_1):
		x1 = np.round(np.random.sample(3))
		x2 = np.random.sample(3) * 0.2 - 0.1
		x = [i+j for i,j in zip(x1,x2)]

		n = int(x1[0])*4 + int(x1[1])*2 + int(x1[2])
		y = [0.0 for k in range(0,8)]
		y[n] = 1.0

		data_set_1.add( Instance(x, y) )	

	data_set_1.saveToFile()

#data_set_1.printInstances()

#Building data set 4

data_set_4 = DataSet("data_set_4")

if (os.path.isfile(data_set_4.name)):
	data_set_4.loadFile()

else:
	data_set_size_4 = 10
	for i in range(0, data_set_size_4):
		t = 2 * np.pi * np.random.sample(1)[0]
		u = np.random.sample(1)[0] + np.random.sample(1)[0]
		r = 2-u if u > 1 else u
		x = [r*np.cos(t), r*np.sin(t)]

		n = 0
		if(x[0] >= 0.0):
			n = 0 if x[1] >= 0.0 else 3
		else:
			n = 1 if x[1] >= 0.0 else 2

		n += 4 if np.absolute(x[1]) > 1.0 - np.absolute(x[0]) else 0

		y = [0.0 for k in range(0,8)]
		y[n] = 1.0

		data_set_4.add( Instance( x, y) )

	data_set_4.saveToFile()

#data_set_4.printInstances()
	
############################################################################
#	Running Neural Network
############################################################################

#Parameters
sig_func = ActivationFunction("sigmoid")
step_func = ActivationFunction("step")

#Creating the Neural Network
net = NeuralNetwork(3, [8], [step_func], 0.1)
net.trainDataSet(data_set_1, "estochastic", 10, momentum = 0.5, print_info = True, type = "class")
