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

def print2DArray(X):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			print(X[i][j],end=" ")
		print("")

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
	LAMBDA = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95]
	FRACTION = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	ERROR = []
	iterations = 10
	print("Lambda\tFraction\tTraining_Error\tTest_Error")
	for fraction in FRACTION:
		err_train = [0]*len(LAMBDA)
		err_test = [0]*len(LAMBDA)
		for itr in range(iterations):
			X = copy.deepcopy(INPUT_X)
			Y = copy.deepcopy(INPUT_Y)
			shuffle(X,Y,fraction)
			standardization(X,fraction)
			NUMBER_OF_ROWS_IN_TRAINING = int(NUMBER_OF_ROWS*fraction)
			NUMBER_OF_ROWS_IN_TEST = NUMBER_OF_ROWS - NUMBER_OF_ROWS_IN_TRAINING
			for l in range(len(LAMBDA)):
				Lambda = LAMBDA[l]
				W = mylinridgereg(X[:NUMBER_OF_ROWS_IN_TRAINING],Y[:NUMBER_OF_ROWS_IN_TRAINING],Lambda)
				Train_Y = mylinridgeregeval(X[:NUMBER_OF_ROWS_IN_TRAINING],W)
				Test_Y = mylinridgeregeval(X[NUMBER_OF_ROWS_IN_TRAINING:],W)

				train_error = 0
				for i in range(len(Train_Y)):
					train_error += ((Train_Y[i]-Y[i])**2)

				test_error = 0
				for i in range(NUMBER_OF_ROWS_IN_TEST):
					test_error += ((Test_Y[i]-Y[i+NUMBER_OF_ROWS_IN_TRAINING])**2)
				
				train_error /= (2*len(Train_Y))
				test_error /= (2*len(Test_Y))

				err_train[l] += train_error
				err_test[l] += test_error
		
		for l in range(len(LAMBDA)):
			print(LAMBDA[l],end="\t")
			print(fraction,end="\t")
			print(float(err_train[l]/iterations),end="\t")
			print(float(err_test[l]/iterations))

	print("--- %s seconds ---" % (time.time() - start_time))
