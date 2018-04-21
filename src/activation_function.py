import numpy as np
from keras.layers import Activation

'''
Funções definidas para permitir uso direto, sem instanciações por parte da
aplicação principal
'''

step_func = Activation("hard_sigmoid")
sig_func = Activation("sigmoid")
tanh_func = Activation("tanh")
relu_func = Activation("relu")