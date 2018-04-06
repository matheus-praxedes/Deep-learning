from perceptron import Perceptron
import numpy as np

class Layer:
	
	def __init__(self, neuron_count, input_count, activation_function):
		self.perceptron_list = [Perceptron(input_count, activation_function) for i in range(neuron_count)]
		
	def process(self, input_signal):
		for neuron in self.perceptron_list:
			neuron.process(input_signal) # função process() da classe Perceptron,

	def updateGradients(self, error_list):
		for index, neuron in enumerate(self.perceptron_list):
			neuron.updateLocalGradient(error_list[index])

	def weightAdjustment(self, learning_rate, momentum):
	#Faz os ajustes dos pesos sinápticos para os neurônios de uma determinada camada.
		for neuron in self.perceptron_list:
			neuron.weightAdjustment(learning_rate, momentum) # Função weightAdjustment() da classe Perceptron.

	def getOutput(self):
		 return [n.getOutput() for n in self.perceptron_list]

	def getSum(self, index_i):
		temp = [j.getLocalGradient() * j.getWeight(index_i+1) for j in self.perceptron_list]
		return np.sum(temp)

	def getSums(self):
		return [self.getSum(i) for i in range(len(self.perceptron_list[0].weight_list) - 1)]

	def getGradients(self):
		return [n.getLocalGradient() for n in self.perceptron_list]