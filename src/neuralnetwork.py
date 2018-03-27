import numpy as np

class Perceptron:
	"""docstring for ClassName"""
	
	v = 0.0
	output = 0.0
	derived_function = 0.0
	local_gradient = 0.0
	activation_function = 0.0
	input_list = []
	weight_list = []
	old_weight_list = []
	
	def __init__(self, number_input, activation_function, derived_function):
		
		weight_list = np.random.random_sample(number_input)
		old_weight_list = []
		
		self.activation_function = activation_function
		self.derived_function = derived_function
	
	def getOutput(self):
		
		return output

	def process(self, input_list):
		
		input_list = [1.0] + input_list 
		self.input_list = input_list
		mul = np.multply(input_list, weight_list)
		v = np.sum(mul)

		output = activation_function(v)

	def getWeight(self, index):
		
		return weight_list[index]

	def getLocalGradient(self):
		
		return local_gradient
	
	def updateLocalGradient(self, sum):
		
		return derived_function(v)*sum

	def weightAdjustment(self, learning_rate):
		
		old_weight_list = weight_list
		weight_list = np.sum(weight_list, np.multply(learning_rate*local_gradient, input_list))  


		
		