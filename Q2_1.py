import matplotlib.pyplot as plt

X_0 = []
Y_0 = []
X_1 = []
Y_1 = []

with open("credit.txt") as f:
	for line in f:
		X = line[:-1].split(",")
		if X[2]=='0':
			X_0.append(float(X[0]))
			Y_0.append(float(X[1]))
		else:
			X_1.append(float(X[0]))
			Y_1.append(float(X[1]))

plt.plot(X_0,Y_0,'y^',linewidth=2.0,markersize=8,label='Credit Card not Issued')
plt.plot(X_1,Y_1,'bo',linewidth=2.0,label='Credit Card Issued')
plt.xlabel('Value of attribute X1',fontsize=12)
plt.ylabel('Value of attribute X2',fontsize=12)
plt.title('Plot of Data', fontsize=14, fontweight='bold')
plt.legend()


plt.show()
