class Instance:
	def __init__(self, input, expected_output = []):
		self.input = input
		self.expected_output = expected_output


class DataSet:
	def __init__(self, instances = []):
		self.instances = instances
