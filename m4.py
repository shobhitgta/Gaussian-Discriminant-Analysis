import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from numpy.linalg import inv
import pandas as pd
import math

def cal_phi(Y):
	m = len(Y)*1.0
	ans = sum(Y == 1)[0]
	return (1.0*ans)/m

def cal_mu1(X,Y):
	ans = np.dot(Y.transpose(), X).transpose()
	ans = ans/(sum(Y==1)[0])
	return ans

def cal_mu0(X,Y):
	Y_temp = 1.0 - Y
	ans = np.dot(Y_temp.transpose(), X).transpose()
	ans = ans/(sum(Y==0)[0])
	return ans

def cal_sigma(X,Y, mu0, mu1):
	sigma = np.zeros((2,2))
	for i in range(0,len(Y)):
		temp = np.zeros((2,1))
		temp[0,0] = X[i,0]
		temp[1,0] = X[i,1]
		if(Y[i,0] == 1):
			diff = np.subtract(temp, mu1)
			update = np.dot(diff, diff.transpose())
			sigma = np.add(sigma, update)
		else:
			diff = np.subtract(temp, mu0)
			update = np.dot(diff, diff.transpose())
			sigma = np.add(sigma, update)
	sigma = np.divide(sigma, len(Y))
	return sigma

def cal_sigma1(X,Y,mu1):
	sigma1 = np.zeros((2,2))
	#print(np.mean(X[:,1]))
	for i in range(0,len(Y)):
		if(Y[i,0] == 1):
			temp = np.zeros((2,1))
			temp[0,0] = X[i,0]
			temp[1,0] = X[i,1]
			diff = np.subtract(temp, mu1)
			#print(diff)
			update = np.dot(diff, diff.transpose())
			#print(update)
			sigma1 = np.add(sigma1, update)
			#print(sigma1)
	sigma1 = sigma1/(sum(Y==1)[0])
	return sigma1

def cal_sigma0(X,Y,mu0):
	sigma0 = np.zeros((2,2))
	for i in range(0,len(Y)):
		if(Y[i,0] == 0):
			temp = np.zeros((2,1))
			temp[0,0] = X[i,0]
			temp[1,0] = X[i,1]
			diff = np.subtract(temp, mu0)
			update = np.dot(diff, diff.transpose())
			sigma0 = np.add(sigma0, update)
	sigma0 = sigma0/(sum(Y==0)[0])
	return sigma0

## function to get the equation for decision boundary
def decision_boundary(x, y, phi, mu0, mu1, sigma0, sigma1):
	Z = np.zeros(x.shape)
	for i in range(0, x.shape[0]-1):
		for j in range(0, x.shape[1]-1):
			X = np.matrix([x[i,j],y[i,j]])
			X = X.transpose()
			diff1 = np.subtract(X, mu1)
			diff0 = np.subtract(X, mu0)
			first = np.dot(inv(sigma1), diff1)
			first = np.dot(diff1.transpose(), first)[0,0]
			#print(inv(sigma0))
			second = np.dot(inv(sigma0), diff0)
			second = np.dot(diff0.transpose(), second)[0,0]
			constant = math.log(phi*1.0/(1.0 - phi))
			det1 = np.linalg.det(sigma1)
			det0 = np.linalg.det(sigma0)
			constant = constant - math.log(math.sqrt((det1 * 1.0)/det0))
			constant = 2 * constant
			Z[i,j] = (first - second - constant)
	return Z

## function to plot decision boundary
def plot_decision_boundary(phi, mu0, mu1, sigma0, sigma1, flag):
	delta = 0.25
	xrange = np.arange(-4, 4, delta)
	yrange = np.arange(-4, 4, delta)
	x, y = np.meshgrid(xrange,yrange)
	z = decision_boundary(x,y, phi, mu0, mu1, sigma0, sigma1)
	if(flag == 0):
		cs = plt.contour(x, y, z, [0], colors='green',linestyles='dashed')
		labels = ['Linear Decision Boundary']
		cs.collections[0].set_label(labels[0])
	else:
		cs = plt.contour(x, y, z, [0])
		labels = ['Quadratic Decision Boundary']
		cs.collections[0].set_label(labels[0])

def main():

	## Reading data
	file = open('q4x.dat')
	X = np.loadtxt(file, delimiter= '  ', skiprows=0)
	Y = np.genfromtxt('q4y.dat', dtype='str').reshape(100,1)
	Y = (Y == 'Alaska').astype(float)
	print(len(Y))
	
	##Data Preprocessing
	for i in range(0,2):
		mean = np.mean(X[:,i]);
		var = np.var(X[:,i]);
		X[:,i] = (X[:,i]-mean)/math.sqrt(var);

	##Calculating Parameters values
	phi = cal_phi(Y)
	mean1 = cal_mu1(X,Y)
	mean0 = cal_mu0(X,Y)
	sigma1 = cal_sigma1(X,Y, mean1)
	sigma0 = cal_sigma0(X,Y, mean0)
	sigma = cal_sigma(X,Y, mean0, mean1)

	## Plot decision Boundary
	plot_decision_boundary(phi, mean0, mean1, sigma0, sigma1, 1)
	plot_decision_boundary(phi, mean0, mean1, sigma, sigma, 0)

	## Plot data
	df = pd.DataFrame(dict(A=X[:,0].transpose(),
                       B=X[:,1].transpose(),
                       C=Y.transpose().flatten()))

	plt.scatter(df.A[df.C == 1], df.B[df.C == 1], c='red', marker='o', label = 'Alaska')
	plt.scatter(df.A[df.C == 0], df.B[df.C == 0], c='aqua', marker='D', label = 'Canada')
	plt.xlabel('growth ring diameters in fresh water')
	plt.ylabel('growth ring diameters in marine water')
	plt.title('Gaussian Discriminant Analysis')
	plt.axis([-3,3,-3,3])
	legend = plt.legend(loc='lower right', fontsize='small')
	plt.show()


if __name__ == "__main__":
    main()