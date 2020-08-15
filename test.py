from libs.NeuralNetwork import  NeuralNetwork  #importing Neural Network Module
from libs.DenseLayer import Layer_Dense  #Dense Layer used to add Layer
from libs.Tensor import Tensor #Tensor used to create ND Tensor

from libs.ActivationFunctions import Act_RELU,Act_Sigmoid,Act_Tanh,Act_Unit,Act_SoftMax #importing Activation Functions

data=Tensor([[1,2,3,2.5],
	[2.0,5.0,-1.0,2.0],
	[-1.5,2.7,3.3,-0.8]]) #creating an Tensor

nnt=NeuralNetwork() #Create an Neaural Network Object
#adding Layers
nnt. addLayer(Layer_Dense(4,5,act_func=Act_RELU))
nnt. addLayer(Layer_Dense(5,2,act_func=Act_Sigmoid))
nnt. addLayer(Layer_Dense(2,5,act_func=Act_Tanh))
nnt. addLayer(Layer_Dense(5,3,act_func=Act_Unit))
nnt. addLayer(Layer_Dense(3,2,act_func=Act_SoftMax))

#Trainig Data
out=nnt.fit_data(data)

#printing Output
print(out)

#Drawing Network
print(nnt.draw_network())