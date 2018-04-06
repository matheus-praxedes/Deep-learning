from layer import Layer
import numpy as np

class NeuralNetwork:
	
	def __init__(self, input_size, layer_size_list, activation_function_list):
		self.layer_list = [Layer(layer_size_list[i], layer_size_list[i-1], activation_function_list[i]) for i in range(1, len(layer_size_list))]
		self.layer_list = [Layer(layer_size_list[0], input_size, activation_function_list[0])] + self.layer_list
		self.layer_size_list = layer_size_list
		self.learning_rate = 0.1
		self.momentum = 0.0
		self.last_input = []
		self.output = []
		self.confusion_matrix = [[]]

	def classify(self, input_signal):
	# Dado os valores de entrada/sinais, a rede é processada e retorna o sinal de saída.
		signal = input_signal
		self.last_input = signal

		for layer in self.layer_list:
			layer.process(signal)
			signal = layer.getOutput()

		self.output = signal
		return signal	

	def getOutputError(self, expected_output):
		# Erro na saída do neurônio para um dado exemplo - slide 07-Backpropagation
		return np.subtract(expected_output, self.output)

	def getInstantError(self, expected_output):
		# Erro instantâneo para todos os neurônios da camada de saída para um dado
		# exemplo - slide 07-Backpropagation.
		x = self.getOutputError(expected_output) 		
		mul = np.multiply(x, x)
		soma = 0.5 * np.sum(mul)
		return soma

	def backpropagation(self, output_error):
		num_layers = len(self.layer_size_list)
		output_layer = num_layers-1
		input_layer = 0

		self.layer_list[output_layer].updateGradients(output_error)
		self.layer_list[output_layer].weightAdjustment(self.learning_rate, self.momentum)

		for layer_id in range(output_layer-1, input_layer-1, -1):
			self.layer_list[layer_id].updateGradients(self.layer_list[layer_id+1].getSums())
			self.layer_list[layer_id].weightAdjustment(self.learning_rate, self.momentum)

	def correctnessTest(self, expected_output):
		# Teste de corretude da rede.
		return self.output == expected_output


	def trainDataSet(self, data_set, training_type, num_epoch = 0, learning_rate = 0.1, momentum = 0.0, mini_batch_size = 10, tvt_ratio = [8, 2, 0], type = "reg", print_info = False):
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		
		# Matriz de confusão.
		self.confusion_matrix = [[0 for x in range(self.layer_size_list[-1])] for y in range(self.layer_size_list[-1])]
		y_axis_train = []
		y_axis_valid = []
		x_axis_epoch = []

		# Definição do tamanho dos subconjuntos de treinamento, validação e teste com base na proporção passada
		# no parâmetro tvt_ratio
		tvt_sum = np.sum(tvt_ratio)
		data_set_size = data_set.size()
		training_set_size = int(data_set_size * tvt_ratio[0] / tvt_sum)
		validation_set_size = int(data_set_size * tvt_ratio[1] / tvt_sum)
		test_set_size = int(data_set_size * tvt_ratio[2] / tvt_sum)
		if(test_set_size == 0):
			test_set_size = 1

		for epoch in range(num_epoch):

			print("\r|| Epoch: {:d} || ".format(epoch+1), end = '')
			error = 0.0
			
			# Loss function
			'''

			Loss functions quantificam o quão perto uma determinada rede neural está do ideal para o qual ela
			está treinando. Para isso, calculamos uma métrica com base no erro que observamos nas previsões da 
			rede. Em seguida, agregamos esses erros em todo o conjunto de dados e calculamos a média deles, e 
			agora temos um único número representativo de quão próximo a rede neural é o seu ideal. Para regressão,
			utilizamos a mse loss, Já para a classificação, utilizamos a hinge loss (que para o nosso caso será 
			calculada com base em um valor de limiar).


			'''
			hinge_error = 0.0

			# Serve para garantir a aleatoriedade dos elementos do treinamento.
			data_set.reorderElements(training_set_size)

			# TRAINING #
			
			# Tipos de treinamentos aceitos: estocástico (stochastic) e por lote (for batch).
			if(training_type == "stochastic"):			
				for obj in data_set.data()[0 : training_set_size]:
					self.classify(obj.input)
					feedback = self.getOutputError(obj.expected_output)
					self.backpropagation(feedback) # backpropagation para cada instância do conjunto de dados.
					error += self.getInstantError(obj.expected_output)
					hinge_error += self.hingeLoss(obj.expected_output)
				error /= training_set_size # erro mse.
				hinge_error /= training_set_size # loss function.

			elif(training_type == "batch"):
				for obj in data_set.data()[0 : training_set_size]:
					self.classify(obj.input)
					error += self.getInstantError(obj.expected_output)
					hinge_error += self.hingeLoss(obj.expected_output)
				error /= training_set_size
				hinge_error /= training_set_size
				self.backpropagation( len(self.output) * [-error] )
				# backpropagation para todas as instâncias do conjunto de dados.
				
			if(type != "reg"):
			# Por padrão, definimos que todas as redes são de regressão. Caso as redes sejam de classificação,
			# deve-se definir explicitamente o tipo classificação no parâmetro type.
				error = hinge_error
			print("Training Error: {:.5f} || ".format(error), end = '') if print_info else 0
			x_axis_epoch.append(epoch)
			y_axis_train.append(error)

			# VALIDATION #
			error = 0.0
			for obj in data_set.data()[training_set_size : training_set_size+validation_set_size]:
				self.classify(obj.input)
				if(type == "reg"):
					error += self.getInstantError(obj.expected_output)
				else:
					error += self.hingeLoss(obj.expected_output)
			error /= validation_set_size
			print("Validation Error: {:.5f} || ".format(error), end = '') if print_info else 0
			y_axis_valid.append(error)

		# TESTING #
		error = 0.0
		for obj in data_set.data()[training_set_size+validation_set_size : data_set_size]:
			self.classify(obj.input)
			if(type == "reg"):
				error += self.getInstantError(obj.expected_output)
			else:
				error += self.hingeLoss(obj.expected_output)
				self.updateConfusionMatrix(obj.expected_output) # Apenas em redes de classificação.
		error /= test_set_size
		print("\n|| Test Error: {:.5f} || \n\n".format(error), end = '') if print_info else 0
	
		
		# Informações apresentadas no terminal - apenas para as redes de classificação.
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

	def hingeLoss(self, expected_output):
		# Cálculo do hinge loss com base em um limiar 0.5 .
		temp = [ 0.0 if i < 0.5 else 1.0 for i in self.output]
		return 0.0 if temp == expected_output else 1.0

	def updateConfusionMatrix(self, expected_output):
		# Matriz de confusão para as redes do tipo classificação.
		temp = [ 0.0 if i < 0.5 else 1.0 for i in self.output]
		classification = temp.index(1.0) if 1.0 in temp else np.random.randint(0, len(expected_output))
		expected_classification = expected_output.index(1.0)

		self.confusion_matrix[expected_classification][classification] += 1


	def getPercentError(self):
		# Utilizado na apresentação das informações das redes de classificação.

		total = 0
		correct = 0
		incorrect = 0

		for index_l, line in enumerate(self.confusion_matrix):
			for index_c, value in enumerate(line):
				total += value
				correct += value if index_l == index_c else 0
				incorrect += 0 if index_l == index_c else value

		return [ 100*correct/total, 100*incorrect/total ]