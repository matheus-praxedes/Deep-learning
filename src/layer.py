from perceptron import Perceptron
import numpy as np

class Layer:
	
	def __init__(self, neuron_count, input_count, activation_function):
		self.perceptron_list = [Perceptron(input_count, activation_function) for i in range(neuron_count)]
		
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
		return [self.getSum(i) for i in range(len(self.perceptron_list[0].weight_list))]

	def updateGradients(self, error_list):
		for neuron, i in zip(self.perceptron_list, range(len(self.perceptron_list))):
			neuron.updateLocalGradient(error_list[i])

	def getGradients(self):
		return [n.getLocalGradient() for n in self.perceptron_list]

	def weightAdjustment(self, learning_rate, momentum):
		for n in self.perceptron_list:
			n.weightAdjustment(learning_rate, momentum)