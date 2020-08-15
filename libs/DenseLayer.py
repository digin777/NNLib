import numpy as np
np.random.seed(0)
class Layer_Dense:
	def __init__(self,n_inputs,n_neurons,act_func=None):
		self.weight=0.10*np.random.randn(n_inputs,n_neurons)
		self.biases=np.zeros((1,n_neurons))
		self.act_func=act_func
	def forward(self,inputs):
		self.output=np.dot(inputs,self.weight) + self.biases
		if self.act_func:
			self.act_func=self.act_func(self.output)
			self.output=self.act_func.Forward()
		return self.output
	def add_act_func(self,act_func):
		self.act_func=act_func
