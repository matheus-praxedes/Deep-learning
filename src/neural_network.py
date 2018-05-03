import numpy as np
np.random.seed(11403723)
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.layers import Dropout

class NeuralNetwork:
	
	'''
	Construtor
	@input_size: número de entradas da rede;
	@layer_size_list: lista com a quantidade de neurônios em cada camada da rede;
	@activation_function_list: lista com funções de ativação para cada camada;
	@seed: seed usada na geração dos pesos dos neurônios.
	'''
	def __init__(self, input_size, layer_size_list, activation_function_list, reg = None, reg_param = 0.0, dropout_rate = 0.0):

		self.layer_size_list = layer_size_list
		self.confusion_matrix = [[]]
		self.output = []

		kernel_reg = None
		if(reg == "l1"):
			kernel_reg = regularizers.l1(reg_param)
		elif(reg == "l2"):
			kernel_reg = regularizers.l2(reg_param)
		elif(reg == "l1_l2"):
			kernel_reg = regularizers.l1_l2(reg_param)
		else:
			kernel_reg = None


		self.model = Sequential()
		self.model.add( Dense(units = layer_size_list[0], input_dim = input_size, kernel_regularizer = kernel_reg) )
		self.model.add( activation_function_list[0] )
		for i in range(1, len(layer_size_list)):
			if dropout_rate != 0.0:
				self.model.add( Dropout(dropout_rate) )
			self.model.add( Dense(layer_size_list[i], kernel_regularizer = kernel_reg) )
			self.model.add( activation_function_list[i] )

	'''
	Processa os dados de entrada usando os pesos atuais da rede, gerando uma saída. Apesar do nome,
	esta função também serve em casos de regressão.
	@input_signal: sinal de entrada fornecido à rede.
	'''
	def classify(self, input_signal):
		self.output = self.model.predict( np.array([input_signal]), batch_size = 1, verbose = 0)[0]
		return self.output

	'''
	Faz o treinamento da rede a partir de um conjunto de dados.
	@data_set: conjunto de dados usado no treinamento;
	@training_type: define se o treinamento é estocástico ("stochastic") ou por lote ("batch")
	@num_epoch: número de épocas a serem utilizados no treinamento;
	@lr_: taxa de aprendizagem para este treinamento;
	@momentum_: termo do momento a ser utilizado;
	@mini_batch_size: tamanho dos lotes para o treinamento usando mini-lote;
	@tvt_ratio: lista que representa a proporção desejada entre os conjuntos de treinamento, validação e teste;
	@type: define se a rede deve ser do tipo regressão ("reg") ou classificação ("class");
	@print_info: define se informações devem ser exibidas durante o treinamento.
	'''
	def fit(self,
			data_set,
			training_type,
			num_epoch = 0,	
			mini_batch_size = 10,
			tvt_ratio = [8, 2, 0],
			type = "reg",
			print_info = False,
			loss_ = "mean_squared_error",
			opt = "sgd",
			lr_ = 0.1,
			momentum_ = 0.0,
			decay_ = 0.0,
			rho_ = 0.9,
			epsilon_ = None,
			beta_1_ = 0.9,
			beta_2_ = 0.999,
			schedule_decay_ = 0.004):
		
		# Definição do tamanho dos subconjuntos de treinamento, validação e teste com base na proporção definida
		# no parâmetro tvt_ratio
		tvt_sum = np.sum(tvt_ratio)
		data_set_size = data_set.size()
		training_set_size = int(data_set_size * tvt_ratio[0] / tvt_sum)
		validation_set_size = int(data_set_size * tvt_ratio[1] / tvt_sum)
		test_set_size = int(data_set_size * tvt_ratio[2] / tvt_sum)
		
		# Separação dos conjuntos de dados de acordo com os tamanho definidos
		x_train = np.array( [ data.input for data in data_set.data()[0 : training_set_size] ] )
		y_train = np.array( [ data.expected_output for data in data_set.data()[0 : training_set_size] ] )
		x_val   = np.array( [ data.input for data in data_set.data()[training_set_size : training_set_size + validation_set_size] ] )
		y_val   = np.array( [ data.expected_output for data in data_set.data()[training_set_size : training_set_size + validation_set_size] ] )
		x_test  = np.array( [ data.input for data in data_set.data()[training_set_size + validation_set_size : data_set_size] ] )
		y_test  = np.array( [ data.expected_output for data in data_set.data()[training_set_size + validation_set_size : data_set_size] ] )
	
		verb = 1 if print_info else 0
		val_data = None if validation_set_size == 0 else (x_val, y_val)
		met = [] if type == "reg" else [metrics.categorical_accuracy]
		mini_batch_size = 1 if training_type == "stochastic" else mini_batch_size
		mini_batch_size = training_set_size if training_type == "batch" else mini_batch_size

		# Otimizadores
		
		kernel_opt = None
		if(opt == "sgd"):
			kernel_opt = optimizers.SGD(decay = decay_, lr = lr_, momentum = momentum_)
		elif(opt == "rmsprop"):
			kernel_opt = optimizers.RMSprop(lr = lr_, rho = rho_, epsilon = epsilon_, decay = decay_)
		elif(opt == "adagrad"):
			kernel_opt = optimizers.Adagrad(lr = lr_, epsilon = epsilon_, decay = decay_)
		elif(opt == "adadelta"):
			kernel_opt = optimizers.Adadelta(lr = lr_, rho = rho_, epsilon = epsilon_, decay = decay_)
		elif(opt == "adam"):
			kernel_opt = optimizers.Adam(lr = lr_, beta_1 = beta_1_, beta_2 = beta_2_, epsilon = epsilon_, decay = decay_)
		elif(opt == "adamax"):
			kernel_opt = optimizers.Adamax(lr = lr_, beta_1 = beta_1_, beta_2 = beta_2_, epsilon = epsilon_, decay = decay_)
		elif(opt == "nadam"):
			kernel_opt = optimizers.Nadam(lr = lr_, beta_1 = beta_1_, beta_2 = beta_2_, epsilon = epsilon_, schedule_decay = schedule_decay_)
	

		# Executando a rede
		
		self.model.compile(loss = loss_,
						   optimizer = kernel_opt,
						   metrics = met )

		info = self.model.fit(x_train, y_train,
							  epochs = num_epoch,
							  batch_size = mini_batch_size,
							  verbose = verb,
							  validation_data = val_data,
							  shuffle = True )

		if test_set_size > 0:	
		
			test_results = self.model.evaluate(x_test, y_test, 
							    				batch_size = mini_batch_size,
							    			    verbose = 1)
			
			self.generateConfusionMatrix(x_test, y_test)
			print(test_results)
		
		x_axis_epoch = [i+1 for i in range(num_epoch)]	

		if(type == "reg"):
			if(validation_set_size > 0):
				return [x_axis_epoch, info.history['loss'], info.history['val_loss'] ]
			else:
				return [x_axis_epoch, info.history['loss'] ]
		else:
			if(validation_set_size > 0):
				return [x_axis_epoch, 
					   [1.0 - i for i in info.history['categorical_accuracy'] ],
				   	   [1.0 - i for i in info.history['val_categorical_accuracy'] ] ]
			else:
				return [x_axis_epoch, 
					   [1.0 - i for i in info.history['categorical_accuracy'] ] ]

	
	def generateConfusionMatrix(self, x_test, y_test):

		self.confusion_matrix = [ [ 0 for x in y_test[0] ] for y in y_test[0] ]
		num_outputs = len(y_test[0])

		for input_, output_ in zip(x_test, y_test):
			predict = self.classify(input_)
			max_value = max(predict)
			class_index = 0
			correct_index = 0

			for k in range(num_outputs):
				if predict[k] == max_value:
					class_index = k

			for k in range(num_outputs):
				if output_[k] == 1.0:
					correct_index = k

			self.confusion_matrix[correct_index][class_index] += 1

	def printMatrix(self):

		print()
		for line in self.confusion_matrix:
			print("| ", end = '')
			for value in line:
				print("{:3d} ".format(value), end = '')
			print(" |")
			



			
