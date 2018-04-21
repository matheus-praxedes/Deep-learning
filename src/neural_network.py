import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
from keras import metrics

class NeuralNetwork:
	
	'''
	Construtor
	@input_size: número de entradas da rede;
	@layer_size_list: lista com a quantidade de neurônios em cada camada da rede;
	@activation_function_list: lista com funções de ativação para cada camada;
	@seed: seed usada na geração dos pesos dos neurônios.
	'''
	def __init__(self, input_size, layer_size_list, activation_function_list, seed = None):
		
		self.layer_size_list = layer_size_list
		self.learning_rate = 0.1
		self.momentum = 0.0
		self.confusion_matrix = [[]]
		self.output = []

		self.model = Sequential()
		self.model.add( Dense(units = layer_size_list[0], input_dim = input_size) )
		self.model.add( activation_function_list[0] )
		for i in range(1, len(layer_size_list)):
			self.model.add( Dense(layer_size_list[i]) )
			self.model.add( activation_function_list[i] )

	'''
	Processa os dados de entrada usando os pesos atuais da rede, gerando uma saída. Apesar do nome,
	esta função também serve em casos de regressão.
	@input_signal: sinal de entrada fornecido à rede.
	'''
	def classify(self, input_signal):
		self.output = self.model.predict( input_signal, batch_size = 1, verbose = 0)
		return self.output

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
	def fit(self, data_set, training_type, num_epoch = 0, learning_rate = 0.1, momentum = 0.0, mini_batch_size = 10, tvt_ratio = [8, 2, 0], type = "reg", print_info = False):
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		verb = 1 if print_info else 0
		met = [] if type == "reg" else [metrics.categorical_accuracy]
		
		# Inicialização da matriz de confusão com zeros
		self.confusion_matrix = [[0 for x in range(self.layer_size_list[-1])] for y in range(self.layer_size_list[-1])]

		# Definição do tamanho dos subconjuntos de treinamento, validação e teste com base na proporção definida
		# no parâmetro tvt_ratio
		tvt_sum = np.sum(tvt_ratio)
		data_set_size = data_set.size()
		training_set_size = int(data_set_size * tvt_ratio[0] / tvt_sum)
		validation_set_size = int(data_set_size * tvt_ratio[1] / tvt_sum)
		test_set_size = int(data_set_size * tvt_ratio[2] / tvt_sum)
		test_set_size = 1 if test_set_size == 0 else test_set_size

		x_train = np.array( [ data.input for data in data_set.data()[0 : training_set_size] ] )
		y_train = np.array( [ data.expected_output for data in data_set.data()[0 : training_set_size] ] )
		x_val   = np.array( [ data.input for data in data_set.data()[training_set_size : training_set_size + validation_set_size] ] )
		y_val   = np.array( [ data.expected_output for data in data_set.data()[training_set_size : training_set_size + validation_set_size] ] )
		x_test  = np.array( [ data.input for data in data_set.data()[training_set_size + validation_set_size : data_set_size] ] )
		y_test  = np.array( [ data.expected_output for data in data_set.data()[training_set_size + validation_set_size : data_set_size] ] )

		if(training_type == "stochastic"):
			mini_batch_size = 1
		elif(training_type == "batch"):
			mini_batch_size = training_set_size


		self.model.compile(loss = losses.mean_squared_error,
						   optimizer = optimizers.SGD(lr = self.learning_rate, momentum = self.momentum),
						   metrics = met )

		info = self.model.fit(x_train, y_train,
							  epochs = num_epoch,
							  batch_size = mini_batch_size,
							  verbose = verb,
							  validation_data = (x_val, y_val),
							  shuffle = True )

		loss_and_metrics = self.model.evaluate(x_test, y_test, 
											   batch_size = mini_batch_size,
											   verbose = verb)

		x_axis_epoch = [i for i in range(num_epoch)]		
		if(type == "reg"):
			return [x_axis_epoch, info.history['loss'], info.history['val_loss'] ]
		else:
			return [x_axis_epoch, 
				   [1.0 - i for i in info.history['categorical_accuracy'] ],
				   [1.0 - i for i in info.history['val_categorical_accuracy'] ] ]