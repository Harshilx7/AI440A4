from matplotlib import image
from utilities import *
from NN import trainNN


def main():
	data = image.imread('test_image.jpg')
	data = np.array(data)
	numCentroid = 5  # 5 classes for k-means

	# Find centroids by calling train
	centroids = train(data, numCentroid)
	centroidList = []
	for c in centroids:
		centroidList.append(c.color.tolist())

	#basicAgent(data, numCentroid, centroids)

	#improved agent using NN
	trainNN(270, 3, 0.001, 9, centroidList, data, 10)


if __name__ == '__main__':
	main()
