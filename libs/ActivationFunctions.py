import numpy as np
class Act_RELU:
	def __init__(self,inputs):
		self.inputs=inputs
		self.name="ReLu"

	def Reluactivation(self,inputs):
		return np.maximum(0,inputs)

	def Forward(self):
		self.output=self.Reluactivation(self.inputs)
		return self.output

class Act_SoftMax:
	def __init__(self,inputs):
		self.inputs=inputs
		self.name="SoftMax"
	def Softnaxactivation(self,x):
		return np.exp(x) / np.sum(np.exp(x))
	def Forward(self):
		self.output=self.Softnaxactivation(self.inputs)
		return self.output

class Act_Sigmoid:
	def __init__(self,inputs):
		self.inputs=inputs
		self.name="Sigmoid"
	def SigmoidActivation(self,x):
		return 1/(1+np.exp(-x))
	def Forward(self):
		self.output=self.SigmoidActivation(self.inputs)
		return self.output


class Act_Unit:
	def __init__(self,inputs):
		self.inputs=inputs
		self.name="Unit"
	def Unitactivation(self,x):
		y = np.ones_like(x)
		y[x < 0.0] = 0.0
		y[x >= 0.0] = 1
		return y
		
	def Forward(self):
		self.output=self.Unitactivation(self.inputs)
		return self.output

class Act_Tanh:
	def __init__(self,inputs):
		self.inputs=inputs
		self.name="TanH"
	def Tanhactivation(self,x):
		return (2/(1+np.exp(-2*x)))-1
	def Forward(self):
		self.output=self.Tanhactivation(self.inputs)
		return self.output