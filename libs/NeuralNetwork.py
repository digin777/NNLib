class NeuralNetwork:
	"""docstring for NeuralNetwork"""
	def __init__(self):
		self.layers=[]
	def addLayer(self,layer):
		self.layers.append(layer)
	def fit_data(self,data):
		if len(self.layers)==0:
			return print("There is no Layer In NeuralNetwork")
		for layer in self.layers:
			data=layer.forward(data)
		return data
	def draw_network(self):
		for layer in self.layers:
			print(layer.act_func.name," -> ",end="")
