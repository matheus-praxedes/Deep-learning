import numpy as np
import os.path
from data import Instance, DataSet
import matplotlib.pyplot as plt

def initialize_data(name, set_size):
	data_set = DataSet(name)

	if (os.path.isfile(data_set.name)):
		data_set.loadFile()
		if(data_set.size() == set_size):
			return data_set
		else:
			data_set = DataSet(name)
	
	for i in range(set_size):

		if (name == "data_set_1"):
			data_set.add( generateInstance_1() )
		if (name == "data_set_3a"):
			data_set.add( generateInstance_3a() )
		if (name == "data_set_3b"):
			data_set.add( generateInstance_3b() )
		if (name == "data_set_4"):
			data_set.add( generateInstance_4() )
		if (name == "data_set_5"):
			data_set.add( generateInstance_5() )

	data_set.saveToFile()

	return data_set

def generateInstance_1():
	x1 = np.round(np.random.sample(3))
		
	phi = np.random.random() * 2.0 * np.pi
	cos_theta = np.random.random() * 2.0 - 1.0
	theta = np.arccos( cos_theta )
	u = np.random.random()
	
	r = 0.1 * np.cbrt( u )
	x = r * np.sin( theta) * np.cos( phi )
	y = r * np.sin( theta) * np.sin( phi )
	z = r * np.cos( theta )
	x2 = [x, y, z]
	x = [i+j for i,j in zip(x1, x2)]

	n = int(x1[0])*4 + int(x1[1])*2 + int(x1[2])
	y = [0.0 for k in range(0,8)]
	y[n] = 1.0

	return Instance(x, y)


def generateInstance_3a():
	x1 = np.random.randint(0, 2)
	x2 = np.random.randint(0, 2)
	x = [float(x1), float(x2)]
	y = 1.0 if x1 == x2 else 0.0 

	return Instance(x, y)


def generateInstance_3b():
	x = [np.random.random() * 4.0 + 0.001]
	y = [np.sin(x[0] * np.pi) / (x[0] * np.pi)]

	return Instance(x, y)


def generateInstance_4():
	t = 2 * np.pi * np.random.sample(1)[0]
	u = np.random.sample(1)[0] + np.random.sample(1)[0]
	r = 2-u if u > 1 else u
	x = [r*np.cos(t), r*np.sin(t)]

	n = 0
	if(x[0] >= 0.0):
		n = 0 if x[1] >= 0.0 else 3
	else:
		n = 1 if x[1] >= 0.0 else 2

	n += 4 if np.absolute(x[1]) > 1.0 - np.absolute(x[0]) else 0

	y = [0.0 for k in range(0,8)]
	y[n] = 1.0

	return Instance( x, y)


def generateInstance_5():
	n = np.random.randint(0, 101)
	x_old = [np.sin(m) for m in range(n-10, n) ]
	x = [np.sin(n-10+idx + x1 * x1) for idx, x1 in enumerate(x_old)]

	y_old = [np.sin(m) for m in range(n+1, n+4) ]
	y = [np.sin(n+1+idx + y1 * y1) for idx, y1 in enumerate(y_old)]

	return Instance( x, y)


def plot_graph(data, title, xlabel, ylabel, figsizex = 8, figsizey = 8):
	plt.figure(figsize=(figsizex, figsizey))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(data[0], data[1], 'r', label = "Training")
	plt.plot(data[0], data[2], 'b', label = "Validation")
	plt.legend(loc = 'center right', bbox_to_anchor = (1.0, 0.7))

def plot_points(data, title, xlabel, ylabel, figsizex = 4, figsizey = 4, point_size = 2):
	plt.figure(figsize=(figsizex, figsizey))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	data_x = [ d[0] for d in data ]
	data_y = [ d[1] for d in data ]

	plt.plot(data_x, data_y, 'ro', markersize = point_size)