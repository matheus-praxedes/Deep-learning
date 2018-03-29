import numpy as np
from data import Instance
from neural_network import NeuralNetwork
from activation_function import ActivationFunction

############################################################################
#	Auxiliary Functions
############################################################################

############################################################################
#	Running Neural Network
############################################################################

#Parameters
data_set_size = 1000
epoch_number  = 1000
training_set_size = 800
validation_set_size = 100
test_set_size = 100
sig_func = ActivationFunction("sigmoid")

#Building training set
data_set = []
for i in range(0, data_set_size):
	x1 = np.round(np.random.random_sample(3))
	x2 = np.random.random_sample(3) * 0.2 - 0.1
	x = [i+j for i,j in zip(x1,x2)]

	n = int(x1[0]) + int(x1[1])*2 + int(x1[2])*4
	y = [0.0 for k in range(0,8)]
	y[n] = 1.0

	data_set.append( Instance(x, y) )

#Creating the Neural Network
net = NeuralNetwork(3, [8], [sig_func], 0.1)

#Training the Neural Network
for i in range(0, epoch_number):

	print("\r", i+1, "/", epoch_number, end = '')

	net.resetMSE()
	
	for obj in data_set[0 : training_set_size]:
		net.train(obj.input, obj.expected_output)
		#print("\r", i, "/", epoch_number, " | Instant error - training: ", net.getInstantError(obj.expected_output), end = '')
		net.getInstantError(obj.expected_output)

	print("\t MSE - training: ", net.getMSE(), end = '')

	net.resetMSE()

	for obj in data_set[training_set_size : training_set_size+validation_set_size]:
		net.classify(obj.input)
		#print("\r", i, "/", epoch_number, " | Instant error - classify: ", net.getInstantError(obj.expected_output), end = '')	
		net.getInstantError(obj.expected_output)

	print("\t|   MSE - classify: ", net.getMSE(), end = '')

print()	

############################################################################
############################################################################
