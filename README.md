# NNLib - simple Neural Network Framework Created From Scratch
NNLib is a simple Open Souce Neaural Network Framework Created Completely From Scratch 
it is completely written in Python People who instrested in NN can fork me on Git hub 
and you can contibute to Development of our project 

## It support The Following Activation Functions
 - **RELU**
 - **Unit Step**
 - **Sigmoid**
 - **Tanh**
 - **SoftMax**

The Module structure 
```
NNLib
.\libs\
	ActivationFunctions.py
	DenseLayer.py
	NeuralNetwork.py
	Tensor.py
.\README.md
.\test.py
```

### How To Use


```python
from libs.NeuralNetwork import  NeuralNetwork  #importing Neural Network Module
from libs.DenseLayer import Layer_Dense  #Dense Layer used to add Layer
from libs.Tensor import Tensor #Tensor used to create ND Tensor

from libs.ActivationFunctions import Act_RELU,Act_Sigmoid,Act_Tanh,Act_Unit,Act_SoftMax #importing Activation Functions

X=Tensor([[1,0,1,1],
	  [0,0,1,1],
	  [1,1,1,0]]) #creating an Tensor
Y=Tensor([[1,1,1],[1,0,1],[0,1,1]])
nnt=NeuralNetwork() #Create an Neaural Network Object
#adding Layers
nnt. addLayer(Layer_Dense(4,3,act_func=Act_Sigmoid))
#Trainig Data
out=nnt.fit_data(X,Y)
```
Predicting the Output
```python
test=Tensor([[1,0,1,1]])
out=nnt.predict(test)
#printing Output
print(out)
```
```python
[[0.94289458 0.90514702 0.98781461]]
```
### Ploting the Network Using
```python
#Drawing Network
nnt.draw_network()
```
```
Sigmoid
```
### printing Network
```python
#Printing Network
nnt.print_network()
```
```
========== Neural Net ==========
no of input 1 Act Function Sigmoid no of neurons 4
================== ==================
```
### Trainig Process

```
epoach  -> {1483} Loss <-> 0.049105531402673826
epoach  -> {1484} Loss <-> 0.04908684302542737
epoach  -> {1485} Loss <-> 0.049068174697725085
epoach  -> {1486} Loss <-> 0.04904952638463827
epoach  -> {1487} Loss <-> 0.049030898051322006
epoach  -> {1488} Loss <-> 0.04901228966301513
epoach  -> {1489} Loss <-> 0.04899370118503974
epoach  -> {1490} Loss <-> 0.04897513258280106
epoach  -> {1491} Loss <-> 0.04895658382178705
epoach  -> {1492} Loss <-> 0.04893805486756845
epoach  -> {1493} Loss <-> 0.04891954568579809
epoach  -> {1494} Loss <-> 0.04890105624221108
epoach  -> {1495} Loss <-> 0.0488825865026244
epoach  -> {1496} Loss <-> 0.04886413643293635
epoach  -> {1497} Loss <-> 0.04884570599912682
epoach  -> {1498} Loss <-> 0.048827295167256826
epoach  -> {1499} Loss <-> 0.04880890390346789
```