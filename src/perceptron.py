import numpy as np
from activation_function import ActivationFunction

class Perceptron:
	
	def __init__(self, input_count, activation_function):
		self.weight_list = np.random.sample(input_count+1)
		self.old_weight_list = self.weight_list
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

	def weightAdjustment(self, learning_rate, momentum):
		old_delta = np.subtract(self.weight_list, self.old_weight_list)
		self.old_weight_list = self.weight_list
		
		op1 = np.multiply(learning_rate*self.local_gradient, self.input_list)
		op2 = np.multiply(momentum, old_delta)
		
		current_delta = [ i+j for i,j in zip(op1, op2)]
		self.weight_list = [ i+j for i,j in zip(self.weight_list, current_delta)] 