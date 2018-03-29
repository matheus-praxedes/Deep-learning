from layer import Layer
import numpy as np

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