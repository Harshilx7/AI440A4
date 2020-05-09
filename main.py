from matplotlib import image
from utilities import *
from NN import trainNN

def main():
	data = image.imread('test_image.jpg')
	data = np.array(data)
	numCentroid = 20  # 5 classes for k-means
	
	# Find centroids by calling train
	centroids = train(data, numCentroid)
	centroidList = []
	for c in centroids:
		centroidList.append(c.color.tolist())
	#basicAgent(data, numCentroid, centroids)
	np.savetxt('testingData/test1_20.txt',centroidList)
	#improved agent using NN
	#Params: size, layers, rate, iter, centroids, data, useAvg, weights
	trainNN(270, 4, 0.003,10, centroidList,data,False,True)


if __name__ == '__main__':
	main()
