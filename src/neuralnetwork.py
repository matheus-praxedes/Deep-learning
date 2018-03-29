import numpy as np

############################################################################
#	Neural Network
############################################################################

class ActivationFunction:

	def __init__(self, function, derivate):
		self.function = function
		self.derivate = derivate

	def getFunction(self):
		return self.function

	def getDerivate(self):
		return self.derivate

############################################################################

class Perceptron:
	
	def __init__(self, input_count, activation_function):
		self.weight_list = np.random.random_sample(input_count+1)
		self.old_weight_list = []
		self.input_list = []
		self.activation_function = activation_function
		self.local_gradient = 0.0
		self.v = 0.0
		self.output = 0.0
	
	def getOutput(self):
		return self.output

	def process(self, input_list):
		input_list = [1.0] + input_list 
		self.input_list = input_list
		mul = np.multiply(input_list, self.weight_list)
		self.v = np.sum(mul)
		self.output = self.activation_function.getFunction()(self.v)

	def getLocalGradient(self):
		return self.local_gradient
	
	def updateLocalGradient(self, sum):
		self.local_gradient = self.activation_function.getDerivate()(self.v)*sum

	def getWeight(self, index):
		return self.weight_list[index]

	def weightAdjustment(self, learning_rate):
		self.weight_list = [ i+j for i,j in zip(self.weight_list, np.multiply(learning_rate*self.local_gradient, self.input_list))] 


##########################################################################

class Layer:
	
	def __init__(self, neuron_count, input_count, activation_function):
		self.perceptron_list = [Perceptron(input_count, activation_function) for i in range(0, neuron_count)]
		
	def getOutput(self):
		output_list = [n.getOutput() for n in self.perceptron_list]
		return output_list

	def process(self, signal_list):
		for neuron in self.perceptron_list:
			neuron.process(signal_list)

	def getSum(self, index_i):
		temp = [j.getLocalGradient()*j.getWeight(index_i) for j in self.perceptron_list]
		return np.sum(temp)

	def getSums(self):
		return [self.getSum(i) for i in range(0, len(self.perceptron_list[0].weight_list))]

	def updateGradients(self, error_list):
		for neuron, i in zip(self.perceptron_list, range(0, len(self.perceptron_list))):
			neuron.updateLocalGradient(error_list[i])

	def getGradients(self):
		return [n.getLocalGradient() for n in self.perceptron_list]

	def weightAdjustment(self, learning_rate):
		for n in self.perceptron_list:
			n.weightAdjustment(learning_rate)


###########################################################

class NeuralNetwork:
	
	def __init__(self, input_size, layer_size_list, activation_function_list, learning_rate = 0.1, momentum = 0.0):
		self.layer_list = [Layer(layer_size_list[i], layer_size_list[i-1], activation_function_list[i]) for i in range(1, len(layer_size_list))]
		self.layer_list = [Layer(layer_size_list[0], input_size, activation_function_list[0])] + self.layer_list
		self.layer_size_list = layer_size_list
		self.error = 0.0
		self.MSE_count = 0
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.last_input = []
		self.output = []

	def classify(self, input_signal):
		signal = input_signal
		self.last_input = signal

		for layer in self.layer_list:
			layer.process(signal)
			signal = layer.getOutput()

		self.output = signal
		return signal

	def train(self, input_signal, expected_output):
		self.classify(input_signal)
		error = self.getOutputError(expected_output)
		self.backpropagation(error)
		

	def getOutputError(self, expected_output):
		return np.subtract(expected_output, self.output)

	def getInstantError(self, expected_output):
		x = self.getOutputError(expected_output) 		
		mul = np.multiply(x, x)
		soma = 0.5 * np.sum(mul)
		self.error += soma
		self.MSE_count += 1
		return soma

	def getMSE(self):
		return self.error / self.MSE_count

	def resetMSE(self):
		self.error = 0.0
		self.MSE_count = 0

	def backpropagation(self, output_error):
		num_layers = len(self.layer_size_list)
		output_layer = num_layers-1
		input_layer = 0

		self.layer_list[output_layer].updateGradients(output_error)
		self.layer_list[output_layer].weightAdjustment(self.learning_rate)

		for layer_id in range(output_layer-1, input_layer-1, -1):
			self.layer_list[layer_id].updateGradients(self.layer_list[layer_id+1].getSums())
			self.layer_list[layer_id].weightAdjustment(self.learning_rate)

	def print(self):
		formatted_list = ["%.3f"%item for item in self.last_input]
		print("Input\t", formatted_list)

		for layer in self.layer_list:
			for perceptron in layer.perceptron_list:
				formatted_list = ["%.3f"%item for item in perceptron.weight_list]
				print(formatted_list)
				print(perceptron.getLocalGradient())

		formatted_list = ["%.3f"%item for item in self.output]
		print("Output\t", formatted_list)
		print("Error\t", self.error)
		print( )


###########################################################

class Instance:
	def __init__(self, input, expected_output = []):
		self.input = input
		self.expected_output = expected_output

############################################################################
############################################################################

############################################################################
#	Auxiliary Functions
############################################################################

def relu(x):
	return np.fmax(0.0, x)		

def derived_relu(x):
	return 0.0 if x < 0.0 else 1.0	

def sig(x):
	return 1.0 / ( 1.0 + np.exp(-x))		

def derived_sig(x):
	return sig(x) * (1.0 - sig(x))

def degrau(x):
	return 1.0 if x>= 0.0 else 0.0	

def derived_degrau(x):
	pass

############################################################################
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
sig_func = ActivationFunction(sig, derived_sig)

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
