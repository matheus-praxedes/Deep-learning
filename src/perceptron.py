import numpy as np
from activation_function import ActivationFunction

class Perceptron:
	
	# O parâmetro input_count representa o número de entradas da rede.

	def __init__(self, input_count, activation_function):
		self.input_signal = [] #valores passados entre as camadas.
		self.output = 0.0

		self.weight_list = np.random.sample(input_count+1)
		self.old_weight_list = self.weight_list 
		# old_weight_list: armazena os valores dos pesos sinápticos antes de se aplicar 
		# os ajustes nos pesos, visando uma consulta posterior.  
		
		self.activation_function = activation_function
		self.local_gradient = 0.0
		self.v = 0.0 
		# v: valor passado para a função de ativação (somatório dos pesos e as entradas
		# /sinais).

	def process(self, input_signal):
		self.input_signal = [1.0] + input_signal
		# Por questões de simplificações, consideramos o bias como sendo a primeira entra-
		# da do perceptron, com valor constante 1. Sendo assim, ele possui também um peso 
		# (peso do bias), assim como nas outras entradas. 
		self.v = np.dot(self.input_signal, self.weight_list)
		self.output = self.activation_function.getFunction()(self.v)

	def updateLocalGradient(self, sum):
		self.local_gradient = self.activation_function.getDerivate()(self.v)*sum
		
	def weightAdjustment(self, learning_rate, momentum):
		# Ajuste dos pesos sinápticos com base nas fórmulas do slide 07-Backpropagation

		old_delta = np.subtract(self.weight_list, self.old_weight_list)
		self.old_weight_list = self.weight_list
		
		op1 = np.multiply(learning_rate * self.local_gradient, self.input_signal)
		op2 = np.multiply(momentum, old_delta)
		
		current_delta = [ i+j for i,j in zip(op1, op2)]
		self.weight_list = [ i+j for i,j in zip(self.weight_list, current_delta)] 

	def getOutput(self):
		return self.output

	def getLocalGradient(self):
		return self.local_gradient

	def getWeight(self, index):
		return self.weight_list[index]