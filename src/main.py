import numpy as np
from data import Instance, DataSet
from neural_network import NeuralNetwork
from activation_function import ActivationFunction


#Building data set
data_set_size = 1000
data_set = DataSet()
for i in range(0, data_set_size):
	x1 = np.round(np.random.random_sample(3))
	x2 = np.random.random_sample(3) * 0.2 - 0.1
	x = [i+j for i,j in zip(x1,x2)]

	n = int(x1[0]) + int(x1[1])*2 + int(x1[2])*4
	y = [0.0 for k in range(0,8)]
	y[n] = 1.0

	data_set.add( Instance(x, y) )

############################################################################
#	Running Neural Network
############################################################################

#Parameters
sig_func = ActivationFunction("sigmoid")

#Creating the Neural Network
net = NeuralNetwork(3, [8], [sig_func], 0.1)
net.trainDataSet(data_set, "estochastic", 100, print_info = True)
