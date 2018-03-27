import numpy as np

class Perceptron:
	"""docstring for ClassName"""
	
	def __init__(self, number_input, activation_function, derived_function):
		
		self.weight_list = np.random.random_sample(number_input+1)
		self.old_weight_list = []
		self.input_list = []
		
		self.activation_function = activation_function
		self.derived_function = derived_function
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

		self.output = self.activation_function(self.v)

	def getWeight(self, index):
		
		return self.weight_list[index]

	def getLocalGradient(self):
		
		return self.local_gradient
	
	def updateLocalGradient(self, sum):
		
		return self.derived_function(v)*sum

	def weightAdjustment(self, learning_rate):
		
		self.old_weight_list = self.weight_list
		self.weight_list = np.sum(self.weight_list, np.multply(learning_rate*self.local_gradient, self.input_list))  





def relu(x):
	
	return np.fmax(0.0, x)		

def derived_relu():
	
	if x >= 0:
		return 1.0
	else:
		return 0.0	
 		

input_test = [10.0, 8.0, 25.0]
test = Perceptron(3, relu, derived_relu)
test.process(input_test)

soma = 0.0

for x in range(0,3):
    print ("input", input_test[x], "weight", test.getWeight(x+1), "product", input_test[x]*test.getWeight(x+1))
    soma += input_test[x]*test.getWeight(x+1)

soma += test.getWeight(0)

print("Soma: ", soma)
print("Output: ", test.getOutput())
