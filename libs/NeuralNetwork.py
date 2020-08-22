import numpy as np
import pickle
import matplotlib.pyplot as plt
class NeuralNetwork:
	"""docstring for NeuralNetwork"""
	def __init__(self,lr=None):
		if lr is not None:
			self.lr=lr
		else:
			self.lr=0.1
		self.layers=[]
		self.no_layer=0

	def addLayer(self,layer):
		self.no_layer+=1
		layer.id=self.no_layer
		self.layers.append(layer)

	@staticmethod
	def loadModel(model):
		with open(model,"rb") as f:
			object=pickle.load(f)
		return object

	def plotlernig(self):
		ax=plt.axes()
		ax.set(title = "Lernig graph",xlabel = "Epoaches",ylabel = "Loss")
		ax.plot(np.arange(self.epoach),self.loss)
		plt.show()

	def fit_data(self,x,y,epoach):
		self.loss=[]
		self.epoach=epoach
		for iter in range(epoach):
			data=x
			if len(self.layers)==0:
				return print("There is no Layer In NeuralNetwork")
			#print("=======================Forward==========================")
			for layer in self.layers:
				#print("layer_id_outside",layer.id)
				data=layer.forward(data)
			#print(f"\nepoach  -> {iter} \n output \n",data)
			error =y-data
			#print("error\n",error)
			self.loss.append(float(np.mean(np.abs(error))))
			print("epoach  ->",{iter},"Loss <->",np.mean(np.abs(error)))
			l=self.layers.copy()
			l.reverse()
			#print("=======================Backward=======================")
			for layer in l:
				error=layer.backward(error=error,lr=self.lr)
				#print("layer_id",layer.id)
				#print("outside ->",layer.weight.shape)

	def predict(self,data):
		for layer in self.layers:
				data=layer.forward(data)
		return data

	def test(self,x,y):
		self.test_loss=[]
		data=x
		for i in range(len(x)):
			for layer in self.layers:
				data=layer.forward(data)
			error=y[i]-data
			self.test_loss.append(np.mean(np.abs(error)))

	def get_Accurecy(self):
		return (100-self.loss[-1])

	def draw_network(self):
		for layer in self.layers:
			print(layer.act_func.name," -> ",end="")
	def print_network(self):
		print("\n",10*"=","Neural Net",10*"=")
		for layer in self.layers:
			print(layer)
		print(18*"=",18*"=")
	def SaveModel(self,model_name,Path):
		with open(Path+"//"+str(model_name),"wb") as f:
			pickle.dump(self,f)

