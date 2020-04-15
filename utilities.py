import numpy as np


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
	left = y-1 >= 0
	right = y+1 < y_max

	temp = []
	if x-1 >= 0:
		if left: temp.append(data[x-1][y-1])
		temp.append(data[x-1][y])
		if right: temp.append(data[x-1][y+1])
	if temp: cluster.append(temp)
	temp = []
	if left: temp.append(data[x][y-1])
	temp.append(data[x][y])
	if right: temp.append(data[x][y+1])
	if temp: cluster.append(temp)
	temp = []
	if x+1 < x_max:
		if left: temp.append(data[x+1][y-1])
		temp.append(data[x+1][y])
		if right: temp.append(data[x+1][y+1])
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
