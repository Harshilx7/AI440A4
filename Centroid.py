import random
import numpy as np


class Centroid:
	color = np.array([])  # Approximate color of the centroids
	list = np.array([[-1, -1, -1]])  # List of all colors in the centroid
	total = np.array([0, 0, 0])  # All the colors in the list added up

	def __init__(self, color):
		self.color = np.array(color)
		self.total = np.array([0, 0, 0])

	# Adds a color to the list to be saved
	def addToList(self, newColor):
		if np.array_equal(self.list[0], [-1, -1, -1]):
			self.list = np.array([newColor])
			# print(self.list)
			self.total = np.add(self.total, self.list[0])
		else:
			color = np.array([newColor])
			self.list = np.append(self.list, color, axis=0)
			# print(self.list)
			self.total = np.add(self.total, np.array(newColor))

	# Updates the color of the centroid based on the list
	def updateCentroid(self):
		if np.array_equal(self.list[0], [-1, -1, -1]):
			# Nothing was added, change the location of the centroid
			x = random.random() * 255
			y = random.random() * 255
			z = random.random() * 255
			self.color = np.array([x, y, z])
			pass
		else:
			row, col = self.list.shape
			self.color = self.total / row
			self.total = np.array([0, 0, 0])

	# Deletes everything in the list
	def restartList(self):
		self.list = np.array([[-1, -1, -1]])
