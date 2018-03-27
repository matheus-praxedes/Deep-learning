import numpy as np

class Perceptron:
	"""docstring for ClassName"""
	
	output = 0.0
	derived_function
	local_gradient
	activation_function
	weight_list
	old_weight_list
	
	def __init__(self, number_input, activation_function, derived_function):
		
		weight_list = np.random.random_sample(number_input)
		old_weight_list = []
		
		self.activation_function = activation_function
		self.derived_function = derived_function
	
	def getOutput(self):
		return output

	def process(self, input_list):
	    input_list = [1.0] + input_list 
	    np.multply(input_list, weight_list)

	def getWeight(self, index):
		return weight_list[index]

	def getLocalGradient(self):
		return local_gradient
	
	def updateLocalGradient(self, sum):
		pass

	def weightAdjustment(self):
		pass


		
		