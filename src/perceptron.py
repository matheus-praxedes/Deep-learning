import numpy as np
from activation_function import ActivationFunction

class Perceptron:
	
	'''
	Construtor
	@input_count: quantidade de entradas que o neurônio possui
	@activation_function: função de ativação do neurônio
	'''
	def __init__(self, input_count, activation_function):
		self.input_signal = []
		self.output = 0.0
		self.activation_function = activation_function
		self.local_gradient = 0.0

		# Cada neurônio possui um peso extra, que é o peso do bias.
		self.weight_list = np.random.sample(input_count+1)

		# Valores dos pesos referentes a um momento anteriror.
		# É atualizado cada vez que os pesos são recalculados.  
		self.old_weight_list = self.weight_list 

		# Valor passado para a função de ativação (somatório dos pesos 
		# multiplicados pelas entradas).
		self.v = 0.0 

	'''
	Calcula a saída do neurônio com base na entrada fornecida
	@input_signal: lista de valores associados às entradas do neurônio
	'''
	def process(self, input_signal):
		
		# Por questões de simplificação, o bias é considerado como a primeira entrada
		# do perceptron, com valor constante 1.0. Sendo assim, ele possui também um peso 
		# (peso do bias), assim como nas outras entradas. 
		self.input_signal = [1.0] + input_signal
		self.v = np.dot(self.input_signal, self.weight_list)
		self.output = self.activation_function.getFunction()(self.v)

	'''
	Atualiza o gradiente do neurônio.
	@sum: somatório dos produtos entre gradientes e pesos dos neurônios 
		 da camada seguinte
	'''
	def updateLocalGradient(self, sum):
		self.local_gradient = self.activation_function.getDerivate()(self.v)*sum
		
	'''
	Ajusta os pesos das sinapses neurônio usando regra delta
	@learning_rate: taxa de aprendizagem
	@momentum: termo do momento
	'''
	def weightAdjustment(self, learning_rate, momentum):
		old_delta = np.subtract(self.weight_list, self.old_weight_list)
		self.old_weight_list = self.weight_list
		
		op1 = np.multiply(learning_rate * self.local_gradient, self.input_signal)
		op2 = np.multiply(momentum, old_delta)
		
		current_delta = [ i+j for i,j in zip(op1, op2)]
		self.weight_list = [ i+j for i,j in zip(self.weight_list, current_delta)] 

	'''
	Retorna a última saída calculada para o neurônio
	'''
	def getOutput(self):
		return self.output

	'''
	Retorna o valor do gradiente local atual
	'''
	def getLocalGradient(self):
		return self.local_gradient

	'''
	Retorna um dos pesos associados ao neurônio
	@index: indica qual dos pesos deve ser retornado (0 para o peso do bias)
	'''
	def getWeight(self, index):
		return self.weight_list[index]