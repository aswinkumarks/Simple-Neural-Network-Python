import numpy as np


def sigmoid(x):
	return np.exp(x)/(1+np.exp(x))

def delta_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x)) 
	# return x*(1-x)

def tanH(x):
	return np.tanh(x)

def delta_tanH(x):
	return 1.0 - x**2

function_mapping = {'sigmoid':[sigmoid,delta_sigmoid],'tanh':[tanH,delta_tanH]}


class NeuralNetwork:
	def __init__(self,layers,acti_fn='sigmoid'):
		print('Using NOVA Network Backend\n Developed by Aswin Kumar\n')
		if acti_fn not in function_mapping:
			print('Invalid activation Function')
			print('Available Functions are ',function_mapping.keys())
			exit(0)

		self.activation_function = acti_fn
		self.nn_activation,self.delta_nn_activation =  function_mapping[acti_fn]
		self.layers = layers
		self.weights = []
		self.biases = []
		for i in range(1,len(layers)):
			weight = np.random.randn(layers[i], layers[i-1])
			biase = np.random.randn(layers[i], 1)
			self.weights.append(weight)
			self.biases.append(biase)


	def forward(self,inputs):
		acti = np.copy(inputs)
		activations = [acti]
		Z_S = []

		for i in range(len(self.weights)):
			z1 = self.weights[i].dot(acti) + self.biases[i]
			Z_S.append(z1)
			acti = self.nn_activation(z1)
			activations.append(acti)

		return Z_S,activations


	def backpropagation(self,Y,Z_S,activations):
		dw = []  
		db = []
		deltas = [None] * len(self.weights)  

		deltas[-1] = (Y-activations[-1])*self.delta_nn_activation(Z_S[-1])

		for i in reversed(range(len(deltas)-1)):
			deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*self.delta_nn_activation(Z_S[i])        

		batch_size = Y.shape[1]
		db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
		dw = [d.dot(activations[i].T)/float(batch_size) for i,d in enumerate(deltas)]

		return dw, db


	def train(self, x, y, batch_size=10, epochs=100, lr = 0.01):
		for e in range(1,epochs): 
			i=0
			while(i<len(y)):
				x_batch = x[i:i+batch_size]
				y_batch = y[i:i+batch_size]
				i = i+batch_size
				z_s, a_s = self.forward(x_batch)
				dw, db = self.backpropagation(y_batch, z_s, a_s)
				self.weights = [w+lr*dweight for w,dweight in  zip(self.weights, dw)]
				self.biases = [w+lr*dbias for w,dbias in  zip(self.biases, db)]
				if e%1000 == 0:
					print("loss = {}".format(np.linalg.norm(a_s[-1]-y_batch) ),end='\r')

		print('\n')

	def save_weights(self,filename='model_weights.npz'):
		np.savez(filename,layers=self.layers,
			     weights=self.weights,biases=self.biases,acti_fn=[self.activation_function])

	def load_weights(self,filename='model_weights.npz'):
		try:
			npzfile = np.load(filename,allow_pickle=True)
			self.layers = npzfile['layers']
			self.weights = npzfile['weights']
			self.biases = npzfile['biases']
			self.nn_activation,self.delta_nn_activation = function_mapping[npzfile['acti_fn'][0]]

		except:
			print('Error loading weights')
			exit(0)



