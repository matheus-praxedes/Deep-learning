from random import shuffle

class Instance:
	def __init__(self, input, expected_output = []):
		self.input = input
		self.expected_output = expected_output


class DataSet:
	def __init__(self):
		self.instances = []

	def size(self):
		return len(self.instances)

	def add(self, instance):
		self.instances.append(instance)

	def data(self):
		return self.instances

	def reorderElements(self, until):
		shuffle(self.instances[0:until])
