import numpy as np

'''
Define as funções de ativação que podem ser utilizadas para o treinamento 
da rede. As funções de ativação relu, tangente hiperbólica (tanh), 
sigmóide (sigmoid) e degrau (step) podem ser instancias a partir de nomes
passados como parâmetros, mas novas funções podem ser escritas manualmente
caso desejado.
'''
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
		elif(function_name == "tanh"):
			self.function = tanh
			self.derivate = derived_tanh

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

def tanh(x):
	power = np.exp(-x)
	return (1 - power) / (1 + power)

def derived_tanh(x):
	tg = tanh(x)
	return 0.5 * (1.0 - tg * tg)

'''
Funções definidas para permitir uso direto, sem instanciações por parte da
aplicação principal
'''
step_func = ActivationFunction("step")
sig_func = ActivationFunction("sigmoid")
tanh_func = ActivationFunction("tanh")
relu_func = ActivationFunction("relu")