from layer import Layer
import numpy as np

class NeuralNetwork:
	
	def __init__(self, input_size, layer_size_list, activation_function_list, learning_rate = 0.1, momentum = 0.0):
		self.layer_list = [Layer(layer_size_list[i], layer_size_list[i-1], activation_function_list[i]) for i in range(1, len(layer_size_list))]
		self.layer_list = [Layer(layer_size_list[0], input_size, activation_function_list[0])] + self.layer_list
		self.layer_size_list = layer_size_list
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.last_input = []
		self.output = []
		self.confusion_matrix = [[]]

	def classify(self, input_signal):
		signal = input_signal
		self.last_input = signal

		for layer in self.layer_list:
			layer.process(signal)
			signal = layer.getOutput()

		self.output = signal
		return signal	

	def getOutputError(self, expected_output):
		return np.subtract(expected_output, self.output)

	def getInstantError(self, expected_output):
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

	def correctnessTest(self, expected_output, threshold = 0.5):
		normalized_output = [1.0 if n > threshold else 0.0 for n in self.output]
		return normalized_output == expected_output


	def trainDataSet(self, data_set, training_type, num_epoch = 0, learning_rate = 0.1, momentum = 0.0, mini_batch_size = 10, tvt_ratio = [9, 1, 1], print_info = False):
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.confusion_matrix = [[0 for x in range(self.layer_size_list[-1])] for y in range(self.layer_size_list[-1])] 

		tvt_sum = np.sum(tvt_ratio)
		data_set_size = data_set.size()
		training_set_size = int(data_set_size * tvt_ratio[0] / tvt_sum)
		validation_set_size = int(data_set_size * tvt_ratio[1] / tvt_sum)
		test_set_size = int(data_set_size * tvt_ratio[2] / tvt_sum)

		for epoch in range(num_epoch):

			print("\r|| Epoch: {:d} || ".format(epoch+1), end = '') if print_info else 0
			error = 0.0
			data_set.reorderElements(training_set_size)
			
			# TRAINING #
			if(training_type == "estochastic"):			
				for obj in data_set.data()[0 : training_set_size]:
					self.classify(obj.input)
					feedback = self.getOutputError(obj.expected_output)
					self.backpropagation(feedback)
					error += self.getInstantError(obj.expected_output)
				error /= training_set_size

			elif(training_type == "batch"):
				for obj in data_set.data()[0 : training_set_size]:
					self.classify(obj.input)
					error += self.getInstantError(obj.expected_output)
				error /= training_set_size
				self.backpropagation( len(self.output) * [-error] )

			elif(training_type == "mini-batch"):
				for batch in range(training_set_size // mini_batch_size):
					error = 0.0
					for obj in data_set.data()[mini_batch_size*batch : mini_batch_size*(batch+1)]:
						self.classify(obj.input)
						error += self.getInstantError(obj.expected_output)
					error /= mini_batch_size
					self.backpropagation( len(self.output) * [-error] )

			print("Training Error: {:.5f} || ".format(error), end = '') if print_info else 0

			# VALIDATION #
			error = 0.0
			for obj in data_set.data()[training_set_size : training_set_size+validation_set_size]:
				self.classify(obj.input)
				error += self.getInstantError(obj.expected_output)
			error /= validation_set_size
			print("Validation Error: {:.5f} || ".format(error), end = '') if print_info else 0

		# TESTING #
		error = 0.0
		for obj in data_set.data()[training_set_size+validation_set_size : data_set_size]:
			self.classify(obj.input)
			error += self.getInstantError(obj.expected_output)
			self.updateConfusionMatrix(obj.expected_output)
		error /= test_set_size
		print("Test Error: {:.5f} || \n".format(error), end = '') if print_info else 0
		
		for line in self.confusion_matrix:
			print("| ", end = '')
			for value in line:
				print("{:3d} ".format(value), end = '')
			print(" |")

		print()
		percent_error = self.getPercentError()
		print("Correct: {:3.1f}%\nIncorrect: {:3.1f}%".format(percent_error[0], percent_error[1]))


	def updateConfusionMatrix(self, expected_output, threshold = 0.5):

		normalized_output = [1.0 if n > threshold else 0.0 for n in self.output]

		if (normalized_output.count(1.0) != 1) :
			normalized_output[self.output.index(self.output.max())] = 1.0

		classification = normalized_output.index(1.0)
		expected_classification = normalized_output.index(1.0)

		self.confusion_matrix[classification][expected_classification] += 1


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