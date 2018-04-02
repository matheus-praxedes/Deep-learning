import numpy as np

class ActivationFunction:

	def __init__(self, function, derivate):
		self.function = function
		self.derivate = derivate

	def __init__(self, function_name):
		if(function_name == "relu"):
			self.function = relu
			self.derivate = derived_relu
		elif(function_name == "sigmoid"):
			self.function = sig
			self.derivate = derived_sig
		elif(function_name == "step"):
			self.function = step
			self.derivate = derived_step

	def getFunction(self):
		return self.function

	def getDerivate(self):
		return self.derivate

def relu(x):
	return np.fmax(0.0, x)		

def derived_relu(x):
	return 0.0 if x < 0.0 else 1.0	

def sig(x):
	return 1.0 / ( 1.0 + np.exp(-x))		

def derived_sig(x):
	return sig(x) * (1.0 - sig(x))

def step(x):
	return 1.0 if x >= 0.0 else 0.0	

def derived_step(x):
	return 1.0