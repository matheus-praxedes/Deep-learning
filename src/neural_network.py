from layer import Layer
import numpy as np

class NeuralNetwork:
	
	'''
	Construtor
	@input_size: número de entradas da rede;
	@layer_size_list: lista com a quantidade de neurônios em cada camada da rede;
	@activation_function_list: lista com funções de ativação para cada camada.
	'''
	def __init__(self, input_size, layer_size_list, activation_function_list):
		self.layer_list = [Layer(layer_size_list[i], layer_size_list[i-1], activation_function_list[i]) for i in range(1, len(layer_size_list))]
		self.layer_list = [Layer(layer_size_list[0], input_size, activation_function_list[0])] + self.layer_list
		self.layer_size_list = layer_size_list
		self.learning_rate = 0.1
		self.momentum = 0.0
		self.last_input = []
		self.output = []
		self.confusion_matrix = [[]]

	'''
	Processa os dados de entrada usando os pesos atuais da rede, gerando uma saída. Apesar do nome,
	esta função também serve em casos de regressão.
	@input_signal: sinal de entrada fornecido à rede.
	'''
	def classify(self, input_signal):
		signal = input_signal
		self.last_input = signal

		for layer in self.layer_list:
			layer.process(signal)
			signal = layer.getOutput()

		self.output = signal
		return signal	

	'''
	Calcula o erro na saída do neurônio, de acordo com a saída atual da rede e a saída desejada.
	@expected_output: saída desejada para o exemplo usado no último treinamento.
	'''
	def getOutputError(self, expected_output):
		return np.subtract(expected_output, self.output)

	'''
	Calcula o erro instantâneo da saída da rede para um dado exemplo.
	@expected_output: saída desejada para o exemplo usado no último treinamento.
	'''
	def getInstantError(self, expected_output):
		x = self.getOutputError(expected_output) 		
		mul = np.multiply(x, x)
		soma = 0.5 * np.sum(mul)
		return soma

	'''
	Dada a lista de erros na saída da rede, calcula a backpropagation.
	@output_error: sinal de erro na saída na rede.
	'''
	def backpropagation(self, output_error):
		num_layers = len(self.layer_size_list)
		output_layer = num_layers-1
		input_layer = 0

		# Para a camada de saída, o gradiente é diretamente calculado a partir do erro na saída.
		# Após isso, os pesos da camada são ajustados
		self.layer_list[output_layer].updateGradients(output_error)
		self.layer_list[output_layer].weightAdjustment(self.learning_rate, self.momentum)

		# A atualização é repetida para as demais camadas, usando os somatórios das camadas posteriores
		# para atualizar seus gradientes
		# BUG - usar valores dos somatórios ANTES de atualizar os pesos
		for layer_id in range(output_layer-1, input_layer-1, -1):
			self.layer_list[layer_id].updateGradients(self.layer_list[layer_id+1].getSums())
			self.layer_list[layer_id].weightAdjustment(self.learning_rate, self.momentum)
			
	'''
	Faz o treinamento da rede a partir de um conjunto de dados.
	@data_set: conjunto de dados usado no treinamento;
	@training_type: define se o treinamento é estocástico ("stochastic") ou por lote ("batch")
	@num_epoch: número de épocas a serem utilizados no treinamento;
	@learning_rade: taxa de aprendizagem para este treinamento;
	@momentum: termo do momento a ser utilizado;
	@mini_batch_size: tamanho dos lotes para o treinamento usando mini-lote;
	@tvt_ratio: lista que representa a proporção desejada entre os conjuntos de treinamento, validação e teste;
	@type: define se a rede deve ser do tipo regressão ("reg") ou classificação ("class");
	@print_info: define se informações devem ser exibidas durante o treinamento.
	'''
	def trainDataSet(self, data_set, training_type, num_epoch = 0, learning_rate = 0.1, momentum = 0.0, mini_batch_size = 10, tvt_ratio = [8, 2, 0], type = "reg", print_info = False):
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		
		# Inicialização da matriz de confusão com zeros
		self.confusion_matrix = [[0 for x in range(self.layer_size_list[-1])] for y in range(self.layer_size_list[-1])]
		y_axis_train = []
		y_axis_valid = []
		x_axis_epoch = []

		# Definição do tamanho dos subconjuntos de treinamento, validação e teste com base na proporção definida
		# no parâmetro tvt_ratio
		tvt_sum = np.sum(tvt_ratio)
		data_set_size = data_set.size()
		training_set_size = int(data_set_size * tvt_ratio[0] / tvt_sum)
		validation_set_size = int(data_set_size * tvt_ratio[1] / tvt_sum)
		test_set_size = int(data_set_size * tvt_ratio[2] / tvt_sum)
		if(test_set_size == 0):
			test_set_size = 1

		for epoch in range(num_epoch):

			x_axis_epoch.append(epoch)
			print("\r|| Epoch: {:d} || ".format(epoch+1), end = '')
			
			# Loss function
			'''

			Loss functions quantificam o quão perto uma determinada rede neural está do ideal para o qual ela
			está treinando. Para isso, calculamos uma métrica com base no erro que observamos nas previsões da 
			rede. Em seguida, agregamos esses erros em todo o conjunto de dados e calculamos a média deles, e 
			agora temos um único número representativo de quão próximo a rede neural é o seu ideal. Para regressão,
			utilizamos a mse loss, Já para a classificação, utilizamos a hinge loss (que para o nosso caso será 
			calculada com base em um valor de limiar).


			'''
			class_error = 0.0
			ms_error = 0.0

			# Garante a aleatoriedade dos elementos do treinamento
			data_set.reorderElements(training_set_size)

			# TRAINING #
			# Tipos de treinamentos aceitos: estocástico (stochastic) e por lote (batch).
			if(training_type == "stochastic"):			
				for obj in data_set.data()[0 : training_set_size]:
					self.classify(obj.input)
					feedback = self.getOutputError(obj.expected_output)
					self.backpropagation(feedback) # backpropagation para cada instância do conjunto de dados.

					# Atualiza o erro adequado dependendo do problema
					if(type == "reg"):
						ms_error += self.getInstantError(obj.expected_output)
					else:
						class_error += self.verifyClassification(obj.expected_output)

				ms_error /= training_set_size # erro médio quadrático (só é usado em regressões)
				class_error /= training_set_size # erro de classificação (só é usado para classificações)

			elif(training_type == "batch"):
				for obj in data_set.data()[0 : training_set_size]:
					self.classify(obj.input)

					# Atualiza o erro médio quadrático sempre, já que tanto na regressão quanto na classificação
					# ele é usado para alimentar o backpropagation. O erro de classificação só é atualizado se 
					# o problema for de classificação, já que não é usado no backpropagation
					ms_error += self.getInstantError(obj.expected_output)
					if(type != "reg"):
						class_error += self.verifyClassification(obj.expected_output)

				ms_error /= training_set_size # erro médio quadrático sempre usado
				class_error /= training_set_size # erro de classificação (só é usado para classificações)
				# backpropagation para todas as instâncias do conjunto de dados
				self.backpropagation( len(self.output) * [ms_error] )
				
			error = ms_error if type == "reg" else class_error
			print("Training Error: {:.5f} || ".format(error), end = '') if print_info else 0
			y_axis_train.append(error)

			# VALIDATION #
			ms_error = 0.0
			class_error = 0.0
			for obj in data_set.data()[training_set_size : training_set_size+validation_set_size]:
				self.classify(obj.input)
				if(type == "reg"):
					ms_error += self.getInstantError(obj.expected_output)
				else:
					class_error += self.verifyClassification(obj.expected_output)

			ms_error /= validation_set_size
			class_error /= validation_set_size

			error = ms_error if type == "reg" else class_error
			print("Validation Error: {:.5f} || ".format(error), end = '') if print_info else 0
			y_axis_valid.append(error)

		# TESTING #
		ms_error = 0.0
		class_error = 0.0
		for obj in data_set.data()[training_set_size+validation_set_size : data_set_size]:
			self.classify(obj.input)
			if(type == "reg"):
				ms_error += self.getInstantError(obj.expected_output)
			else:
				class_error += self.verifyClassification(obj.expected_output)
				self.updateConfusionMatrix(obj.expected_output) # Apenas em redes de classificação.

		ms_error /= test_set_size
		class_error /= test_set_size

		error = ms_error if type == "reg" else class_error
		print("\n|| Test Error: {:.5f} || \n\n".format(error), end = '') if print_info else 0
	
		
		# Exibição da matriz de confusão e da porcentagem de acerto para as redes de classificação
		if(type != "reg"):
			print()
			for line in self.confusion_matrix:
				print("| ", end = '')
				for value in line:
					print("{:3d} ".format(value), end = '')
				print(" |")

			print()
			percent_error = self.getPercentError()
			print("Correct: {:3.1f}%\nIncorrect: {:3.1f}%".format(percent_error[0], percent_error[1]))

		return [x_axis_epoch, y_axis_train, y_axis_valid]

	'''
	Verifica se a saída da rede está correta, considerando que o problema é de classificação.
	Primeiro, o valor de saída atual é convertido para 0's e 1's, com o zero representado
	neurônios não-ativados e 1 representado os ativados. É usado um limiar de 0.5.
	@expected_output: saída da rede desejada para o último exemplo classificado.
	'''
	def verifyClassification(self, expected_output):
		temp = [ 0.0 if i < 0.5 else 1.0 for i in self.output]
		return 0.0 if temp == expected_output else 1.0

	'''
	Atualiza a matriz de confusão para as redes do tipo classificação.
	@expected_output: saída da rede desejada para o último exemplo classificado.
	'''
	def updateConfusionMatrix(self, expected_output):
		temp = [ 0.0 if i < 0.5 else 1.0 for i in self.output]
		classification = temp.index(1.0) if 1.0 in temp else np.random.randint(0, len(expected_output))
		expected_classification = expected_output.index(1.0)

		self.confusion_matrix[expected_classification][classification] += 1

	'''
	Calcula e retorna uma lista contendo a porcentagem de erros e acertos para uma rede 
	que resolve problemas de classificação. Só deve ser usada caso a matriz de confusão 
	já tenha sido construída, já utiliza os valores dela.
	'''
	def getPercentError(self):

		total = 0
		correct = 0
		incorrect = 0

		for index_l, line in enumerate(self.confusion_matrix):
			for index_c, value in enumerate(line):
				total += value
				correct += value if index_l == index_c else 0
				incorrect += 0 if index_l == index_c else value

		return [ 100*correct/total, 100*incorrect/total ]