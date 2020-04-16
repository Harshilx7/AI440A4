from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import image
from utilities import *


def main():
	data = image.imread('test_image.jpg')
	data = np.array(data)
	numCentroid = 5  # 5 classes for k-means

	# makes the colored picture into grayscale
	# newData = [[gray(y) for y in x]for x in data]
	# newData = np.array(newData)

	# change data to newData if changing from colored to grayscale
	cluster = getCluster((1, 1), data)
	print(cluster)
	cluster = getAverageOfCluster(cluster)
	print(cluster)
	train(data, numCentroid)
	print("Done")
	# plt.imshow(data)
	# plt.show()


if __name__ == '__main__':
	main()
