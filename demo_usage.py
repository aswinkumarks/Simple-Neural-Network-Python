from network import NeuralNetwork
import numpy as np

if __name__ == '__main__':

	# NOT Gate using Neural Network
	X = np.array([[1,0,1,0]])
	Y = np.array([[0,1,0,1]])
	nn = NeuralNetwork(layers=[1,4,1])
	# print(nn.W1,nn.W2)
	nn.train(X,Y,batch_size=2,epochs=100000, lr = .1)
	# nn.save_weights()
	# nn.load_weights()
	print('Prediction:')
	for i in range(len(X[0])):
		print('Desired Value ',Y[0][i])
		z_s,a_s = nn.forward(X[0][i])
		print('Actual',a_s[-1])
    # nn = NeuralNetwork([1, 100, 1])
    # X = 2*np.pi*np.random.rand(1000).reshape(1, -1)
    # y = np.sin(X)
    # print(X.shape,y.shape)
    
    # nn.train(X, y, epochs=10000, batch_size=64, lr = .1)
    # _, a_s = nn.feedforward(X)
    # print(y, a_s)