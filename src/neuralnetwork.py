import numpy as np

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

	def getWeight(self, index):
		
		return self.weight_list[index]

	def getLocalGradient(self):
		
		return self.local_gradient
	
	def updateLocalGradient(self, sum):
		
		return self.activation_function.getDerivate()(self.v)*sum

	def weightAdjustment(self, learning_rate):
		
		self.old_weight_list = self.weight_list
		self.weight_list = np.sum(self.weight_list, np.multply(learning_rate*self.local_gradient, self.input_list)) 


############################################################################ 

class Layer:	
	def getOutput(self):
		pass

	def process(self, signal_list):
		pass

	def getSum(self, index_j):
		pass

	def getSums(self):
		pass

	def updateGradients(self, error_list):
		pass

	def getGradients(self):
		pass

	def weightAdjustment(self, learning_rate):
		pass


##########################################################################

class HiddenLayer(Layer):
	
	def __init__(self, neuron_count, input_count, activation_function):
		
		self.perceptron_list = [Perceptron(input_count, activation_function) for i in range(0, neuron_count)]
		
	def getOutput(self):
		
		output_list = [n.getOutput() for n in self.perceptron_list]
		return output_list

	def process(self, signal_list):
		
		for i in self.perceptron_list:
			i.process(signal_list)

	def getSum(self, index_j):

		temp = [n.getLocalGradient()*n.getWeight(index_j) for n in self.perceptron_list]
		return np.sum(temp)

	def getSums(self):
		return [self.getSum(i) for i in range(0, len(self.perceptron_list))]

	def updateGradients(self, error_list): #Output layer only
		for neuron, i in zip(self.perceptron_list, range(0, len(self.perceptron_list))):
			neuron.updateLocalGradient(error_list[i])

	def getGradients(self):
		return [n.getLocalGradient() for n in self.perceptron_list]

	def weightAdjustment(self, learning_rate):
		for n in perceptron_list:
			n.weightAdjustment(learning_rate)


############################################################################

class InputLayer(Layer):
	
	def __init__(self, neuron_count):
		
		self.perceptron_list = [Perceptron(1, ActivationFunction()) for i in range(0, neuron_count)]
		
	def getOutput(self):
		
		output_list = [n.getOutput() for n in self.perceptron_list]
		return output_list

	def process(self, signal_list):
		
		for i in range(0, len(self.perceptron_list)):
			self.perceptron_list[i].process([signal_list[i]])

###########################################################


class NeuralNetwork:
	
	def __init__(self, layer_size_list, activation_function_list, learning_rate = 0.1, momentum = 0.0):

		self.layer_list = [HiddenLayer(layer_size_list[i], layer_size_list[i-1], activation_function_list[i-1]) for i in range(1, len(layer_size_list))]
		self.layer_list = [InputLayer(layer_size_list[0])] + self.layer_list
		self.layer_size_list = layer_size_list
		self.error = []
		self.learning_rate = learning_rate
		self.momentum = momentum

	def classify(self, input_signal):

		signal = input_signal

		for layer in self.layer_list:
			layer.process(signal)
			signal = layer.getOutput()

		return signal

	def trainny(self, input_signal, correct_output):

		self.updateOutputError(input_signal, correct_output)
		self.backpropagation(self.error)


	def updateOutputError(self, input_signal, correct_output):
		output = self.classify(input_signal)
		self.error = np.subtract(correct_output, output)


	def backpropagation(self, output_error):

		num_layers = len(self.layer_size_list)
		output_layer = num_layers-1
		input_layer = 0

		self.layer_list[output_layer].updateGradients(output_error)

		for layer_id in range(output_layer-1, input_layer-1, -1):
			self.layer_list[layer_id].updateGradients(self.layer_list[layer_id+1].getSums())
			self.layer_list[layer_id].weightAdjustment(self.learning_rate)


def relu(x):
	
	return np.fmax(0.0, x)		

def derived_relu(x):
	
	if x >= 0:
		return 1.0
	else:
		return 0.0	


relu_func = ActivationFunction(relu, derived_relu)
#test = Layer(5, 3, relu_func)
input_test = [0.02, 1.01, 0.95]
#test.process(input_test)

#print(test.getSum(2))

net = NeuralNetwork([3, 8], [relu_func])
net.trainny(input_test, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#print([1.0, 0.4] + [1.1])