import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Centroid import *


# Find distance between two rgb values. This is the Euclidean distance squared
def calcDistance(rgb1, rgb2):
	x1, y1, z1 = rgb1[0], rgb1[1], rgb1[2]
	x2, y2, z2 = rgb2[0], rgb2[1], rgb2[2]

	return (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2


# Returns the initial random values for the centroids by [r,g,b]
def randomSelection(numCentroid):
	i = 0
	centroidsDict = {}
	centroids = []
	while i < numCentroid:
		x = random.random() * 255
		y = random.random() * 255
		z = random.random() * 255
		if (x, y, z) not in centroidsDict:
			centroidsDict[(x, y, z)] = True

			c = Centroid([x, y, z])  # new Centroid object
			centroids.append(c)

			i += 1
	return centroids


def findClosestCentroid(color, centroids, dictionary):
	strColor = np.array2string(color)
	if strColor in dictionary:
		dictionary[strColor].addToList(color)  # This is the minimum centroid already
		return dictionary[strColor]

	minDist = calcDistance(centroids[0].color, color)
	minCentroid = centroids[0]

	for i in range(1, len(centroids)):
		dist = calcDistance(centroids[i].color, color)

		if dist < minDist:
			minDist = dist
			minCentroid = centroids[i]

	dictionary[strColor] = minCentroid
	minCentroid.addToList(color)


def updateCentroidList(centroids):
	for i in range(len(centroids)):
		centroids[i].updateCentroid()


def printCentroidList(centroids):
	for i in range(len(centroids)):
		print(centroids[i].color)


def restartList(centroids):
	for i in range(len(centroids)):
		centroids[i].restartList()


def train(data, numCentroid, borderSize):
	row, col, _ = data.shape
	# removes borders from calculations
	row = row - (borderSize*2)
	col = col - (borderSize*2)
	train_data = data[:, :int(col / 2)]
	col = int(col / 2)  # New col for the split image

	centroids = randomSelection(numCentroid)

	iterations = 100  # Number of times to run k-means: TODO: determine by elbow method
	for it in range(iterations):
		print("Iteration " + str(it))
		dictionary = {}

		# First, we find the closest centroid to each point in the training set
		for i in range(borderSize, row):
			for j in range(borderSize, col):
				findClosestCentroid(train_data[i][j], centroids, dictionary)

		# Second, we update the centroid values
		updateCentroidList(centroids)

		printCentroidList(centroids)
		plotCurrIteration(centroids)

		restartList(centroids)




def lossFunction():
	pass


def getAverageOfCluster(cluster):
	x, y, z = 0, 0, 0
	for i in range(len(cluster)):
		for j in range(len(cluster[i])):
			x += cluster[i][j][0]
			y += cluster[i][j][1]
			z += cluster[i][j][2]

	size = len(cluster) * len(cluster[0])
	return [round(x / size), round(y / size), round(z / size)]


#
# adds border around the picture
#
# data: pixels of photo
# clusterDim: cluster dimension
#
def addBorder(data, clusterDim=3):
	white = [255, 255, 255]

	border = [white for i in range(int(np.ceil(clusterDim/2))-1)]
	border = np.reshape(border, (-1, 1, 3))
	data = np.array(data)

	# inserts borders
	data = np.insert(data, 0, border, axis=0)
	data = np.insert(data, 0, border, axis=1)
	data = np.insert(data, len(data), border, axis=0)
	data = np.insert(data, len(data[0]), border, axis=1)

	return data


#
# gives a 3x3 cluster with the given
# pixel in the center
#
# coordinate: location of pixel
# data: pixels of photo
# clusterDim: cluster dimension
#
def getCluster(coordinate, data, clusterDim=3):
	row, col = coordinate
	borderSize = int(np.ceil(clusterDim/2)) - 1
	cluster = []
	[[cluster.append(data[row-i][col+j-borderSize]) for j in range(0, clusterDim)] for i in range(borderSize, 0, -1)]
	[[cluster.append(data[row+i][col+j-borderSize]) for j in range(0, clusterDim)] for i in range(0, borderSize+1)]

	cluster = np.array(cluster)
	cluster = np.reshape(cluster, (clusterDim, clusterDim, 3))

	return cluster


#
# converts image pixel to grayscale
#
# rgb: rgb values of the pixel
#
def gray(rgb):
	r, g, b = rgb
	n = int((0.21 * r) + (0.72 * g) + (0.07 * b))
	return [n] * 3


def plotCurrIteration(centroids):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for i in range(len(centroids)):

		color = centroids[i].color / 255
		color = (color[0], color[1], color[2])
		l = centroids[i].list
		if np.array_equal(l[0], [-1, -1, -1]):
			continue
		row, col = l.shape
		r = tuple(l[:, 0].tolist())
		g = tuple(l[:, 1].tolist())
		b = tuple(l[:, 2].tolist())
		ax.scatter(r, g, b, c=color)

		ax.set_xlabel('R')
		ax.set_ylabel('G')
		ax.set_zlabel('B')

	plt.show()