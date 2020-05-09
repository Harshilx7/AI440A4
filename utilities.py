from bisect import bisect_right
from copy import deepcopy
import matplotlib.pyplot as plt
from Centroid import *


# Find distance between two rgb values. This is the Euclidean distance squared
def calcDistance(rgb1, rgb2, weights):
	x1, y1, z1 = rgb1[0], rgb1[1], rgb1[2]
	x2, y2, z2 = rgb2[0], rgb2[1], rgb2[2]

	wX, wY, wZ = 1, 1, 1

	if weights:
		wX = 0.21
		wY = 0.72
		wZ = 0.07

	return (wX * (x1 - x2) ** 2) + (wY * (y1 - y2) ** 2) + (wZ * (z1 - z2) ** 2)


# Finds distance without taking the square and accounting for weights. Only used to calculate the final accuracy
def calcBasicDistance(rgb1, rgb2):
	x1, y1, z1 = rgb1[0], rgb1[1], rgb1[2]
	x2, y2, z2 = rgb2[0], rgb2[1], rgb2[2]

	return (x1 - x2) + (y1 - y2) + (z1 - z2)


# Returns the initial random values for the centroids by [r,g,b]
def randomSelection(numCentroid):
	i = 5
	centroidsDict = {}
	centroids = []
	if numCentroid >= 5:  # Initizlies the first 5 centroids to be extreme colors: Red, Blue, Green, Black, White
		centroids.append(Centroid([255, 0, 0]))  # Red
		centroids.append(Centroid([0, 255, 0]))  # Green
		centroids.append(Centroid([0, 0, 255]))  #
		centroids.append(Centroid([255, 255, 255]))
		centroids.append(Centroid([0, 0, 0]))
		centroidsDict[(255, 0, 0)] = True
		centroidsDict[(0, 255, 0)] = True
		centroidsDict[(0, 0, 255)] = True
		centroidsDict[(255, 255, 255)] = True
		centroidsDict[(0, 0, 0)] = True
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


# Finds closest centroid given a color
def findClosestCentroid(color, centroids, dictionary, addToList):
	strColor = np.array2string(color)
	if strColor in dictionary:
		dictionary[strColor].addToList(color)  # This is the minimum centroid already
		return dictionary[strColor]

	minDist = calcDistance(centroids[0].color, color, not addToList)
	minCentroid = centroids[0]

	for i in range(1, len(centroids)):
		dist = calcDistance(centroids[i].color, color, not addToList)

		if dist < minDist:
			minDist = dist
			minCentroid = centroids[i]

	if addToList:
		dictionary[strColor] = minCentroid
		minCentroid.addToList(color)

	return minCentroid


# Updates the centroid list
def updateCentroidList(centroids):
	for i in range(len(centroids)):
		centroids[i].updateCentroid()


# Prints all colors in the centroid
def printCentroidList(centroids):
	for i in range(len(centroids)):
		print(centroids[i].color)


# Deletes every saved color in each centroid. Doesn't delete the color itself
def restartList(centroids):
	for i in range(len(centroids)):
		centroids[i].restartList()


# Training the k-means
def train(data, numCentroid):
	row, col, _ = data.shape
	train_data = data[:, :int(col / 2)]
	col = int(col / 2)  # New col for the split image

	centroids = randomSelection(numCentroid)

	iterations = 10  # Number of times to run k-means
	print("Finding centroids with " + str(iterations) + " iterations")
	for it in range(iterations):
		print("Iteration " + str(it + 1))
		dictionary = {}

		# First, we find the closest centroid to each point in the training set
		for i in range(row):
			for j in range(col):
				findClosestCentroid(train_data[i][j], centroids, dictionary, True)

		# Second, we update the centroid values
		updateCentroidList(centroids)
		# if (it + 1) % 10 == 0:
		# 	plotCurrIteration(centroids)
		# 	printCentroidList(centroids)

		restartList(centroids)

	print("Finished Finding Centroids")
	# plotCurrIteration(centroids)
	return centroids


def basicAgent(data, numCentroids, centroids):
	# Save a copy of the data
	copy = np.array(deepcopy(data))

	# Start coloring the Test Data
	colorTestData(data, centroids)

	# Get accuracy by comparing the correct centroid with the predicted centroid on the testing side
	loss = calcLoss(copy, data, centroids)
	accuracy = (768 - loss) / 768

	# Plot the updated image
	plt.title("Basic Agent: With Weights\nNumber of Centroids: {0} Loss: {1} Accuracy: {2}".format(str(numCentroids),
																								   str(loss),
																								   str(accuracy)))
	plt.imshow(data)
	plt.show()
	return centroids


# Calculating loss by comparing what the centroid should be vs what centroid was chosen
def calcLoss(copy, data, centroids):
	print("Calculating Accuracy")
	row, col, _ = copy.shape
	val = 0
	for i in range(row):
		for j in range(int(col / 2), col):
			col1 = findClosestCentroid(copy[i][j], centroids, {}, False).color
			col1 = [int(col1[0]), int(col1[1]), int(col1[2])]
			col2 = data[i][j]

			val += calcBasicDistance(col1, col2)

	return round(val / (row * (int(col / 2))), 2)


# Starts the coloring process
def colorTestData(data, centroids):
	print("Converting data to gray scale")
	gray_scale_image, train_dict = convertToGrayScale(data)
	print("Finished Converting to Gray Scale")

	row, col, _ = data.shape
	gray_train = gray_scale_image[:, :int(col / 2)]
	gray_test = gray_scale_image[:, int(col / 2):]

	print("Converting testing data from gray scale to color")
	convertGrayToColor(gray_train, gray_test, centroids, train_dict, data)
	print("Finished converting testing data from gray scale to color")


# Starts the recoloring process on the testing side
def convertGrayToColor(gray_train, gray_test, centroids, train_dict, data):
	# Sorted version of training data. Allows for quick access to the 6 closest grayscale colors
	sortedTrainData = np.sort(gray_train.flatten())

	row, col = gray_test.shape

	for i in range(row):
		for j in range(col):
			# Get 6 closest colors to gray_test[i][j] on the training side
			colors = get6ClosestColors(sortedTrainData, gray_test[i][j], train_dict, data)

			# Get the color of the most common centroid
			centroid_color = getApproxColorUsing6Closest(colors, centroids)

			# Now set the value in data to this new color
			data[i][j + col] = centroid_color  # j + col since col is the size of the train and test data


# Finds 6 gray colors on the Training side that is most similar to newGray
def get6ClosestColors(sortedTrainData, newGray, train_dict, data):
	ind = np.searchsorted(sortedTrainData, newGray)
	size = int(sortedTrainData.shape[0])

	left, right = 0, 0  # Used as the boundaries to receive 7 closest. 7 because of indexing issues
	if ind - 4 < 0:
		left = 0
		right = 7
	elif ind + 3 > size:
		left = size - 7
		right = size
	else:
		left = ind - 4
		right = ind + 3

	# Will take every element from left to right of ind and put it in here.
	# The reason for cornerArr is because multiple colors can have the same gray scale values. If they do have the same
	# gray scale values, it is stored in train_dict. CornerArr stores the index of the training data that corresponds
	# to the gray scale color and it will store the gray scale color.
	cornerArr = []
	for i in range(left, right):
		if sortedTrainData[i] in train_dict:  # Should always be the case
			indArr = train_dict[sortedTrainData[i]]
			for j in range(len(indArr)):
				gray_color = sortedTrainData[i]
				indexOfColor = indArr[j]
				cornerArr.append([gray_color, indexOfColor])
		else:
			print("sortedTrainData[i] is not in train_dict")

	sorted(cornerArr, key=lambda val: val[0])  # Sorts by the color again

	finalColors = []
	keys = [r[0] for r in cornerArr]
	newInd = bisect_right(keys, newGray)

	size = newInd
	if newInd - 3 < 0:
		left = 0
		right = 6
	elif newInd + 3 > size:
		left = size - 6
		right = size
	else:
		left = newInd - 3
		right = newInd + 3

	for i in range(left, right):
		# The row and col of the location of the color in the train_data
		x = cornerArr[i][1][0]
		y = cornerArr[i][1][1]

		color = data[x][y]
		finalColors.append(color)

	return finalColors


# Gets the best color using the 6 closest centroids
def getApproxColorUsing6Closest(colors, centroids):
	cent_dict = {}
	for i in range(len(colors)):
		cent = findClosestCentroid(colors[i], centroids, {}, addToList=False)
		if cent in cent_dict:
			cent_dict[cent] = cent_dict[cent] + 1
		else:
			cent_dict[cent] = 1

	# Now find the centroid that comes up the most in cent_dict
	maxNum = 0
	max_centroid = centroids[0]
	for key, val in cent_dict.items():
		if val > maxNum:
			maxNum = val
			max_centroid = key

	return max_centroid.color


# Averages the elements of the entire cluster and takes its grayscale value
def getAverageOfCluster(cluster):
	tot = 0
	for i in range(len(cluster)):
		for j in range(len(cluster[i])):
			tot += gray(tuple(cluster[i][j]))

	size = len(cluster) * len(cluster[0])
	return tot / size


# Takes data as a 3d list with rgb values and returns an np array of the same list in gray scale
def convertToGrayScale(data):
	row, col, _ = data.shape

	gray_scale_image = []
	train_dict = {}

	for i in range(row):
		image_row = []

		for j in range(int(col / 2)):
			# cluster = getCluster((i, j), data)
			# val = getAverageOfCluster(cluster)
			val = gray(tuple(data[i][j]))

			if val in train_dict:
				train_dict[val].append([i, j])
			else:
				train_dict[val] = [[i, j]]

			image_row.append(val)

		for j in range(int(col / 2), col):
			# cluster = getCluster((i, j), data)
			# val = getAverageOfCluster(cluster)

			val = gray(tuple(data[i][j]))
			image_row.append(val)

		gray_scale_image.append(image_row)

	return np.array(gray_scale_image), train_dict


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
	n = (0.21 * r) + (0.72 * g) + (0.07 * b)
	return n


# Plots the centroids list. Look at graphs 12 - 16
def plotCurrIteration(centroids):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for i in range(len(centroids)):

		color = centroids[i].color / 255.0
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
