import numpy as np
np.random.seed(0)
class Layer_Dense:
	def __init__(self,n_inputs,n_neurons,act_func=None):
		self.weight=0.10*np.random.randn(n_inputs,n_neurons)
		self.biases=np.zeros((1,n_neurons))
		self.act_func=act_func()

	def forward(self,inputs):
		self.input=inputs
		#print("input->",inputs.shape,"weight->","layer_id",self.id,self.weight.shape,end="")
		self.output=np.dot(inputs,self.weight) + self.biases
		#print(self.weight)
		#print("|")
		if self.act_func:
			self.output=self.act_func.Forward(self.output)
		return self.output

	def backward(self,error,lr):
		self.error=error
		#print("error->",error.shape)
		self.out_delta=self.error*self.sigmoidPrime(self.output)
		#print("delta->\n",self.out_delta)
		layer_error =self.out_delta.dot(self.weight.T)
		#print("layer_error->",layer_error.shape)
		#print("input of this layer\n",self.input)
		x=lr*(self.input.T.dot(self.out_delta))
		self.weight += x
		#print("weight changed\n",x)
		#print("weight->",self.weight.shape)
		return layer_error

	def __str__(self):
		return f"no of input {self.input.shape[0]} Act Function {self.act_func.name} no of neurons {self.input.shape[1]}"

	def add_act_func(self,act_func):
		self.act_func=act_func

	def sigmoidPrime(self,x):
		return x *(1-x)
