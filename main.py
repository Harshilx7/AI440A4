from matplotlib import image
from utilities import *


def main():
	data = image.imread('test_image.jpg')
	data = np.array(data)
	numCentroid = 5  # 5 classes for k-means
	clusterDim = 3  # cluster dimension
	# borderSize = int(np.ceil(clusterDim / 2)) - 1

	# newData = addBorder(data, clusterDim)

	# makes the colored picture into grayscale
	# newData = [[gray(y) for y in x]for x in data]
	# newData = np.array(newData)

	# change data to newData if changing from colored to grayscale
	# cluster = getCluster((2, 2), newData, clusterDim)
	# print(cluster.shape)
	# print(cluster)
	# cluster = getAverageOfCluster(cluster)
	# print(cluster)
	# train(newData, numCentroid, borderSize)
	basicAgent(data, numCentroid)
	# print("Done")
	plt.imshow(data)
	plt.show()


if __name__ == '__main__':
	main()
