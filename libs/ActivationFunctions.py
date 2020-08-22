import numpy as np
class Act_RELU:
	def __init__(self):
		self.name="ReLu"

	def Reluactivation(self,inputs):
		return np.maximum(0,inputs)

	def DerRelu(self,x):
		y = np.ones_like(x)
		y[x < 0.0] = 0.0
		y[x >= 0.0] = 1
		return y

	def Forward(self,inputs):
		self.output=self.Reluactivation(inputs)
		return self.output

	def Backward(self,inputs):
		out=self.DerRelu(inputs)
		return out

class Act_SoftMax:
	def __init__(self):
		self.name="SoftMax"

	def Softnaxactivation(self,x):
		return np.exp(x) / np.sum(np.exp(x))

	def DerSoftmax(self,x):
		pass

	def Forward(self,inputs):
		self.output=self.Softnaxactivation(inputs)
		return self.output

	def Backward(self,inputs):
		out=self.DerSoftmax(inputs)
		return out

class Act_Sigmoid:
	def __init__(self):
		self.name="Sigmoid"

	def SigmoidActivation(self,x):
		return 1/(1+np.exp(-x))

	def DirSigmid(self,x):
		return x * (1-x)

	def Forward(self,inputs):
		self.output=self.SigmoidActivation(inputs)
		return self.output

	def Backward(self,inputs):
		out=self.DirSigmid(inputs)
		return out

class Act_Unit:
	def __init__(self):
		self.name="Unit"

	def Unitactivation(self,x):
		y = np.ones_like(x)
		y[x < 0.0] = 0.0
		y[x >= 0.0] = 1
		return y
		
	def DerUnitstep(self,x):
		y = np.ones_like(x)
		y[x != 0.0] = 0.0
		y[x == 0.0] = np.inf
		return y

	def Forward(self,inputs):
		self.output=self.Unitactivation(inputs)
		return self.output

	def Backward(self,inputs):
		out=self.DerUnitstep(inputs)
		return out

class Act_Tanh:
	def __init__(self):
		self.name="TanH"

	def Tanhactivation(self,x):
		return (2/(1+np.exp(-2*x)))-1

	def DerTanh(self,x):
		return 1 - (self.Tanhactivation(x))**2

	def Forward(self,inputs):
		self.output=self.Tanhactivation(inputs)
		return self.output

	def Backward(self,inputs):
		out=self.DerTanh(inputs)
		return out