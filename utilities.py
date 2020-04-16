import random
import numpy as np
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


def train(data, numCentroid):
	row, col, _ = data.shape
	train_data = data[:, :int(col / 2)]
	col = int(col / 2)  # New col for the split image

	centroids = randomSelection(numCentroid)

	iterations = 100  # Number of times to run k-means: TODO: determine by elbow method
	for it in range(iterations):
		print("Iteration " + str(it))
		dictionary = {}

		# First, we find the closest centroid to each point in the training set
		for i in range(row):
			for j in range(col):
				findClosestCentroid(train_data[i][j], centroids, dictionary)

		# Second, we update the centroid values
		updateCentroidList(centroids)

		printCentroidList(centroids)


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
# gives a 3x3 cluster with the given
# pixel in the center
#
# coordinate: location of pixel
# dimensions: size of the photo in pixels
#
def getCluster(coordinate, data):
	x, y = coordinate
	x_max, y_max, _ = data.shape
	cluster = []
	left = y - 1 >= 0
	right = y + 1 < y_max

	temp = []
	if x - 1 >= 0:
		if left: temp.append(data[x - 1][y - 1])
		temp.append(data[x - 1][y])
		if right: temp.append(data[x - 1][y + 1])
	if temp: cluster.append(temp)
	temp = []
	if left: temp.append(data[x][y - 1])
	temp.append(data[x][y])
	if right: temp.append(data[x][y + 1])
	if temp: cluster.append(temp)
	temp = []
	if x + 1 < x_max:
		if left: temp.append(data[x + 1][y - 1])
		temp.append(data[x + 1][y])
		if right: temp.append(data[x + 1][y + 1])
	if temp: cluster.append(temp)

	return np.array(cluster)


#
# converts image pixel to grayscale
#
# rgb: rgb values of the pixel
#
def gray(rgb):
	r, g, b = rgb
	n = int((0.21 * r) + (0.72 * g) + (0.07 * b))
	return [n] * 3
