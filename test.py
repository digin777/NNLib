from libs.NeuralNetwork import  NeuralNetwork  #importing Neural Network Module
from libs.DenseLayer import Layer_Dense  #Dense Layer used to add Layer
from libs.Tensor import Tensor #Tensor used to create ND Tensor

from libs.ActivationFunctions import Act_RELU,Act_Sigmoid,Act_Tanh,Act_Unit,Act_SoftMax #importing Activation Functions


from sklearn.model_selection import train_test_split
from sklearn import metrics,model_selection
from sklearn import datasets
iris =datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)
print(x_train.shape,y_train.shape)
y_train.shape=(4,30)
'''X=Tensor([[1,2,3,2.5],
	[2.0,5.0,-1.0,2.0],
	[-1.5,2.7,3.3,-0.8]]) #creating an Tensor

Y=Tensor([[2,1,2],[3,1,2],[4,5,2]])'''
X=Tensor([[1,0,1,1],
	  [0,0,1,1],
	  [1,1,1,0]])

Y=Tensor([[1,1,1],[1,0,1],[0,1,1]])

nnt=NeuralNetwork() #Create an Neaural Network Object
#adding Layers
#nnt. addLayer(Layer_Dense(4,3,act_func=Act_RELU))
#nnt. addLayer(Layer_Dense(3,3,act_func=Act_Tanh))
nnt. addLayer(Layer_Dense(4,3,act_func=Act_Sigmoid))
#nnt. addLayer(Layer_Dense(3,3,act_func=Act_Unit))
#nnt. addLayer(Layer_Dense(3,3,act_func=Act_RELU))

#Trainig Data
nnt.fit_data(X,Y,1500)
test=Tensor([[1,0,1,1]])
#printing Output
out=nnt.predict(test)
print(test,"\n",out)

test=Tensor([[0,0,1,1]])
out=nnt.predict(test)
print(test,"\n",out)

test=Tensor([[1,1,1,0]])
out=nnt.predict(test)
print(test,"\n",out)
#nnt.test(x_test,y_test)
nnt.plotlernig()
#nnt.SaveModel("NN","E:")
#Drawing Network
nnt.draw_network()
nnt.print_network()