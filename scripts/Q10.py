import time
start_time = time.time()
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import copy

NUMBER_OF_ROWS = 4177
NUMBER_OF_COLUMNS = 11

INPUT_X = np.zeros((NUMBER_OF_ROWS,NUMBER_OF_COLUMNS), dtype=np.float)
INPUT_Y = np.zeros((NUMBER_OF_ROWS,1), dtype=np.float)

def input(file_path):
	with open(file_path) as f:
		itr = 0
		for line in f:
			T = line[:-1].split(",")
			n = len(T)

			INPUT_X[itr][0] = 1

			if T[0]=='F':
				INPUT_X[itr][1] = 1
			elif T[0]=='I':
				INPUT_X[itr][2] = 1
			else:
				INPUT_X[itr][3] = 1

			for i in range(1,n-1):
				f = float(T[i])
				INPUT_X[itr][i+3] = f

			INPUT_Y[itr] = float(T[-1])

			itr += 1

def standardization(X,fraction):
	n = NUMBER_OF_COLUMNS
	C = [0]*n
	S = [0]*n

	m = int(NUMBER_OF_ROWS*fraction)

	for itr in range(m): 
		for i in range(1,n):
			C[i] += X[itr][i]
			S[i] += X[itr][i]*X[itr][i]

	for i in range(len(C)):
		C[i] /= m
		S[i] /= m
	
	for itr in range(NUMBER_OF_ROWS):
		for i in range(1,n):
			X[itr][i] -= C[i]
			X[itr][i] /= pow(S[i]-C[i]*C[i],0.5)

def shuffle(X,Y,fraction):
	n = int(fraction*X.shape[0])
	for i in range(n):
		j = rand.randint(i,n-1)
		X[[i,j]] = X[[j,i]]
		Y[[i,j]] = Y[[j,i]]

def Identity(n):
	T = np.full((n,n),0)
	for i in range(n):
		T[i][i] = 1
	return np.mat(T)

def mylinridgereg(X,Y,Lambda):
	W = np.mat( np.linalg.inv(np.mat(X.T) * np.mat(X) + Lambda*Identity(X.T.shape[0]) ) )* np.mat(X.T) * np.mat(Y) 
	return W

def mylinridgeregeval(X,W):
	return np.mat(X) * np.mat(W)

if __name__ == "__main__":
	input("../linregdata")
	
	Lamda = 2.5
	fraction = 0.4
	W_Final = [0]*NUMBER_OF_COLUMNS
	itr = 100

	for i in range(itr):
		X = copy.deepcopy(INPUT_X)
		Y = copy.deepcopy(INPUT_Y)	
		
		shuffle(X,Y,fraction)
		standardization(X,fraction)

		NUMBER_OF_ROWS_IN_TRAINING = int(NUMBER_OF_ROWS*fraction)
		NUMBER_OF_ROWS_IN_TEST = NUMBER_OF_ROWS - NUMBER_OF_ROWS_IN_TRAINING

		W = mylinridgereg(X[:NUMBER_OF_ROWS_IN_TRAINING],Y[:NUMBER_OF_ROWS_IN_TRAINING],Lamda)
		for j in range(len(W)):
			W_Final[j] += W[j]
	
	for i in range(len(W)):
		W[i] = W_Final[i]/itr

	Z = mylinridgeregeval(X,W)
	
	A = []
	for i in range(30):
		A.append(i)

	plt.plot(Y[:NUMBER_OF_ROWS_IN_TRAINING],Z[:NUMBER_OF_ROWS_IN_TRAINING],'g+')
	plt.plot(A,A,'b-')
	plt.xlabel("Actual Values")
	plt.ylabel("Predicted Values")
	plt.title("For training dataset with lambda=" + str(Lamda) + " and fraction=" + str(fraction))
	plt.savefig("graphs\\Q11_train.png")
	plt.close()

	plt.plot(Y[NUMBER_OF_ROWS_IN_TRAINING:],Z[NUMBER_OF_ROWS_IN_TRAINING:],'r+')
	plt.plot(A,A,'b-')
	plt.xlabel("Actual Values")
	plt.ylabel("Predicted Values")
	plt.title("For test dataset with lambda=" + str(Lamda) + " and fraction=" + str(fraction))
	plt.savefig("graphs\\Q11_test.png")
	plt.close()

	#############################

	plt.plot(Y[:NUMBER_OF_ROWS_IN_TRAINING],Z[:NUMBER_OF_ROWS_IN_TRAINING],'g+')
	plt.plot(A,A,'b-')
	plt.xlabel("Actual Values")
	plt.ylabel("Predicted Values")
	plt.title("For complete dataset with lambda=" + str(Lamda) + " and fraction=" + str(fraction))
	plt.plot(Y[NUMBER_OF_ROWS_IN_TRAINING:],Z[NUMBER_OF_ROWS_IN_TRAINING:],'r+')
	plt.legend(['Training', 'y=x', 'Test'], loc='upper left')
	plt.savefig("graphs\\Q11_combine.png")



	
