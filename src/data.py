from random import shuffle
import pickle

'''
As seguintes classes estruturam/organizam as instâncias (classe Instance) e o 
conjunto de instâncias (classe DataSet) que serão utilizados nos treiset_typentos/
testes das redes
'''

class Instance:
	def __init__(self, input, expected_output = []):
		self.input = input
		self.expected_output = expected_output

	def printInstance(self):
		print(self.input)
		print(self.expected_output)
		print()

class DataSet:
	def __init__(self):
		self.instances = []

	def size(self):
		return len(self.instances)

	def add(self, instance):
		self.instances.append(instance)

	def data(self):
		return self.instances

	'''
	Reordena de forma aleatórios as until primeiras instâncias do conjunto
	de instâncias
	'''
	def reorderElements(self, until):
		shuffle(self.instances[0:until])

	def printInstances(self):
		for instance in self.instances: 
			instance.printInstance()

