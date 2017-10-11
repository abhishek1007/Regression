import numpy as np
import random
import math
from scipy.special import expit

import matplotlib
from matplotlib import pyplot as plt
import scipy
from sklearn import svm

X=[]
Y=[]
W=[]
Fx=[]

alpha=0.1
lamb=0.0
n=100
max_iterations_gd=8000
max_iterations_nr=7
deg=1


X_0 = []
Y_0 = []
X_1 = []
Y_1 = []

def plot(X,deg,W,method):
        if(deg==1):
                plt.figure()

                title='Learned Decision Boundary for '+method
                
                plt.plot(X_0,Y_0,'y^',linewidth=2.0,markersize=8,label='Credit Card not Issued')
                plt.plot(X_1,Y_1,'bo',linewidth=2.0,label='Credit Card Issued')
                plt.xlabel('Value of attribute X1',fontsize=12)
                plt.ylabel('Value of attribute X2',fontsize=12)
                plt.title(title, fontsize=14, fontweight='bold')
                

                plt.xlim([min(min(X_0),min(X_1))-0.2,max(max(X_0),max(X_1))+0.2])
                plt.ylim([min(min(Y_0),min(Y_1))-0.2,max(max(Y_0),max(Y_1))+0.2])



                plot_x = [min(min(X_0),min(X_1))-0.2,max(max(X_0),max(X_1))+0.2]   
                
                plot_y=[0,0]
                plot_y[0] = (-1/W[2,0])*(W[1,0]*plot_x[0] +W[0,0])
                plot_y[1] = (-1/W[2,0])*(W[1,0]*plot_x[1] +W[0,0])

                plt.plot(plot_x, plot_y,'r-',label='Decision Boundary')
                plt.legend()
                plt.show()            
                

                
        else:

                title='Learned Decision Boundary for '+method+' with feature transformation to degree '+str(deg)
                
                h = .02  # step size in the mesh
                # create a mesh to plot in
                x_min, x_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
                y_min, y_max = X[:, 2].min() - 0.2, X[:, 2].max() + 0.2
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
                
                Z=[[i for i in range(-1, int((x_max-x_min)/h))] for j in range(-1, int((y_max-y_min)/h))]

                for i in range(-1, int((y_max-y_min)/h)):
                        for j in range(-1, int((x_max-x_min)/h)):
                                Z[i][j]=result(xx[i,j],yy[i,j],W,deg)

                plt.contour(xx, yy, Z,levels=[0.50001])#, cmap=plt.cm.Paired)  
                plt.plot(X_0,Y_0,'y^',linewidth=2.0,markersize=8,label='Credit Card not Issued')
                plt.plot(X_1,Y_1,'bo',linewidth=2.0,label='Credit Card Issued')
                plt.xlabel('Value of attribute X1',fontsize=12)
                plt.ylabel('Value of attribute X2',fontsize=12)
                plt.title(title, fontsize=14, fontweight='bold')
                plt.show()       
                                                
                      

def f(X,W):
        K=X*W
        K=K[0,0]
        return expit(K)    

def test(W,deg):
        X=[]
        Y=[]
        with open("credit_test.txt") as ff:
	        for line in ff:
		        I = line[:-1].split(",")
		        X.append([1,float(I[0]),float(I[1])])
		        Y.append([float(I[2])])
        X=np.matrix(X)
        Y=np.matrix(Y)
        X=featuretransform(X,deg,1000)
        accuracy(X,Y,W)


def accuracy(X,Y,W):
        correct=0
        for i in range(0,n):
                Fxi=f(X[i],W)
                if((Fxi>=0.5 and Y[i,0]==1.0) or (Fxi<0.5 and Y[i,0]==0.0)):
                        correct=correct+1;
        print "Accuracy: ",(float(correct)/n)*100,"%"   

def result(x1,x2,W,d):
        X=[1,x1,x2]
        X_new=[1]        
        for i in range(1,d+1):
                for j in range(0,i+1):
                        X_new.append(pow(X[2],j)*pow(X[1],i-j))
        X_new=np.matrix(X_new)  
        
        return f(X_new,W)                     

def gradient_descent(W,X,Fx,Y):
        # Gradient Descent

        W_old=W
        W_new=W
        first=True
        j=0
        
        while(j<max_iterations_gd):

                W_old=W_new
                for i in range(0,n):
                        Fx[i,0]=f(X[i],W_old)
                W_new=W_old*(1-((alpha*lamb)/n))-(alpha/n)*(X.transpose()*(Fx-Y))
                # undoing zeroth one
                W_new[0,0]=W_new[0,0]+W_old[0,0]*(((alpha*lamb)/n))
                j=j+1
        
        print "Gradient Descent on training Data"
        accuracy(X,Y,W_new) 
        return W_new               


def newton_raphson(W,X,Fx,Y,R):
        # Newton Raphson

        W_old=W
        W_new=W
        first=True
        j=0

        while(j<max_iterations_nr):
                '''
                if(first==True):
                        first=False
                '''
                W_old=W_new
                for i in range(0,n):
                        fxi=f(X[i],W_old)
                        Fx[i,0]=fxi
                        R[i,i]=(1-fxi)*fxi
                        
                W_new=W_old-(np.linalg.inv(X.transpose()*R*X+(lamb/n)*np.eye(((deg+1)*(deg+2))/2)))*((X.transpose()*(Fx-Y))+(lamb/n)*W_old)
                # undoing zeroth one
                W_new[0,0]=W_new[0,0]+W_old[0,0]*(((alpha*lamb)/n))
                j=j+1
                
        print "Newton Raphson on training Data"
        accuracy(X,Y,W_new)  
        return W_new

def featuretransform(X,d,n):
        X_new=[]
        for k in range(0,n):
                l=[1]
                for i in range(1,d+1):
                        for j in range(0,i+1):
                                l.append(pow(X[k,2],j)*pow(X[k,1],i-j))
                X_new.append(l)
        X_new=np.matrix(X_new)  
        return X_new                                    

with open("credit.txt") as ff:
	for line in ff:
		I = line[:-1].split(",")
		X.append([1,float(I[0]),float(I[1])])
		Y.append([float(I[2])])
		Fx.append([0.5]);
		if I[2]=='0':
			X_0.append(float(I[0]))
			Y_0.append(float(I[1]))
		else:
			X_1.append(float(I[0]))
			Y_1.append(float(I[1]))



for i in range(0,(deg+1)*(deg+2)/2):
        W.append([0])        

X=np.matrix(X)
Y=np.matrix(Y)
W=np.matrix(W)
Fx=np.matrix(Fx)
R=np.eye(n)


print 
W1=gradient_descent(W,X,Fx,Y);  
          
print "Learned Parameters: "
print W1

print "Gradient Descent On Test Data"
test(W1,deg)

plot(X,deg,W1,'Gradient Descent')

print
print
W2=newton_raphson(W,X,Fx,Y,R);

print "Learned Parameters: "
print W2

print "Newton Raphson On Test Data"
test(W2,deg)

plot(X,deg,W2,'Newton Raphson')
print


