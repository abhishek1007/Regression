import matplotlib.pyplot as plt
from scipy.interpolate import spline

LAMBDA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95]
FRACTION = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

l = len(LAMBDA)
f = len(FRACTION)

Lam = []
Frac = []
Train_E = []
Test_E = []

with open("error_data",'r') as f:
	for line in f:
		T = line[:-1].split("\t")
		n = len(T)
		Lam.append(float(T[0]))
		Frac.append(float(T[1]))
		Train_E.append(float(T[2]))
		Test_E.append(float(T[3]))

itr = 0
n = len(Lam)

for lam in LAMBDA:
	X = []
	Y_Train = []
	for i in range(n):
		if Lam[i]==lam:
			X.append(Frac[i])
			Y_Train.append(Train_E[i])

	plt.plot(X,Y_Train)

	itr += 1

plt.xlabel("Frac----->")
plt.ylabel("Error for training dataset------>")
plt.title("For Training Data with lamda in [0,5]")
plt.savefig("graphs\\train_lam"+".png")
plt.close()

for lam in LAMBDA:
	X = []
	Y_Test = []
	for i in range(n):
		if Lam[i]==lam:
			X.append(Frac[i])
			Y_Test.append(Test_E[i])

	plt.plot(X,Y_Test)

	itr += 1

plt.xlabel("Frac----->")
plt.ylabel("Error for Test dataset------>")
plt.title("For Test Data with lamda in [0,5]")
plt.savefig("graphs\\test_lam"+".png")
plt.close()




############################################################################
# For Training data where lamda is x axis
Legend = []
for itr in range(len(FRACTION)//2):
	frac = FRACTION[itr]
	X = []
	Y_Train = []
	for i in range(n):
		if Frac[i]==frac:
			if(Lam[i]!=0):
				X.append(Lam[i])
				Y_Train.append(Train_E[i])

	plt.plot(X,Y_Train,'+')
	Legend.append("frac="+str(frac))

	itr += 1

plt.xlabel("Lamda----->")
plt.ylabel("Error for Training dataset------>")
plt.title("For Training Data with fraction = [0.1,0.4] ")
plt.legend(Legend, loc='upper left')
plt.savefig("graphs\\train_frac_1"+".png")
plt.close()

Legend = []
for itr in range(len(FRACTION)//2,len(FRACTION)):
	frac = FRACTION[itr]
	X = []
	Y_Train = []
	for i in range(n):
		if Frac[i]==frac:
			if(Lam[i]!=0):
				X.append(Lam[i])
				Y_Train.append(Test_E[i])

	plt.plot(X,Y_Train,'+')
	Legend.append("frac="+str(frac))

	itr += 1

plt.xlabel("Lamda----->")
plt.ylabel("Error for Training dataset------>")
plt.title("For Training Data with fraction = [0.5,0.9] ")
plt.legend(Legend, loc='upper left')
plt.savefig("graphs\\train_frac_2"+".png")
plt.close()

############################################################################
# For Test data where lamda is x axis
Legend = []
for itr in range(len(FRACTION)//2):
	frac = FRACTION[itr]
	X = []
	Y_Test = []
	for i in range(n):
		if Frac[i]==frac:
			if(Lam[i]!=0):
				X.append(Lam[i])
				Y_Test.append(Test_E[i])

	plt.plot(X,Y_Test,'+')
	Legend.append("frac="+str(frac))

	itr += 1

plt.xlabel("Lamda----->")
plt.ylabel("Error for Test dataset------>")
plt.title("For Test Data with fraction = [0.1,0.4] ")
plt.legend(Legend, loc='upper left')
plt.savefig("graphs\\test_frac_1"+".png")
plt.close()

Legend = []
for itr in range(len(FRACTION)//2,len(FRACTION)):
	frac = FRACTION[itr]
	X = []
	Y_Test = []
	for i in range(n):
		if Frac[i]==frac:
			if(Lam[i]!=0):
				X.append(Lam[i])
				Y_Test.append(Test_E[i])

	plt.plot(X,Y_Test,'+')
	Legend.append("frac="+str(frac))

	itr += 1

plt.xlabel("Lamda----->")
plt.ylabel("Error for Test dataset------>")
plt.title("For Test Data with fraction = [0.5,0.9] ")
plt.legend(Legend, loc='upper left')
plt.savefig("graphs\\test_frac_2"+".png")
plt.close()