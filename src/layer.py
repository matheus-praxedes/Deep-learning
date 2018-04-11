from perceptron import Perceptron
import numpy as np

class Layer:
	
	'''
	Construtor
	@neuron_count: número de neurônios na camada
	@input_count: número de entradas que os neurônios terão
	@activation_function: função de ativação a ser usadas pelos neurônios da camada
	'''
	def __init__(self, neuron_count, input_count, activation_function):
		self.perceptron_list = [Perceptron(input_count, activation_function) for i in range(neuron_count)]
		
	'''
	Calcula a saída de todos os neurônios da camada
	@input_signal: lista que representa todas as entradas fornecidas aos neurônios
	'''
	def process(self, input_signal):
		for neuron in self.perceptron_list:
			neuron.process(input_signal)

	'''
	Atualiza o valor dos gradientes de todos os neurônios da camada
	@error_list: uma lista que deve conter os resultados dos somatórios entre pesos 
				 e gradientes de todos os neurônios da camada posterior.
	'''
	def updateGradients(self, error_list):
		for index, neuron in enumerate(self.perceptron_list):
			neuron.updateLocalGradient(error_list[index])

	'''
	Faz os ajustes dos pesos sinápticos de todos os neurônios da camada
	@learning_rate: taxa de aprendizagem
	@momentum: termo do momento
	'''
	def weightAdjustment(self, learning_rate, momentum):
		for neuron in self.perceptron_list:
			neuron.weightAdjustment(learning_rate, momentum)

	'''
	Retorna uma lista com as saídas atuais de todos os neurônios
	'''
	def getOutput(self):
		 return [n.getOutput() for n in self.perceptron_list]

	'''
	Retorna o somatório dos produtos entre gradientes locais e pesos das sinapses 
	que ligam os neurônios desta camada a determinado neurônio da camada anterior
	@index_i: índice que indica em relação a qual neurônio devem ser as sinapses 
			  usadas no cálculo
	'''
	def getSum(self, index_i):
		temp = [j.getLocalGradient() * j.getWeight(index_i+1) for j in self.perceptron_list]
		return np.sum(temp)

	'''
	Retorna uma lista contendos os valores da função getSum em relação a cada um 
	dos neurônios da camada anterior
	'''
	def getSums(self):
		return [self.getSum(i) for i in range(len(self.perceptron_list[0].weight_list) - 1)]

	'''
	Retorna os gradientes de todos os neurônios em uma lista
	'''
	def getGradients(self):
		return [n.getLocalGradient() for n in self.perceptron_list]


	def weightAdjustmentBatch(self, learning_rate, momentum):
		for neuron in self.perceptron_list:
			neuron.weightAdjustmentBatch(learning_rate, momentum)
