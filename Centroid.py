import random

import numpy as np


class Centroid:
	color = np.array([])
	list = np.array([[-1, -1, -1]])
	total = np.array([0, 0, 0])  # All the colors in the list added up

	def __init__(self, color):
		self.color = np.array(color)
		self.total = np.array([0, 0, 0])

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

	def updateCentroid(self):
		if np.array_equal(self.list[0], [-1, -1, -1]):
			# Nothing was added, don't change the location of the cluster
			x = random.random() * 255
			y = random.random() * 255
			z = random.random() * 255
			self.color = np.array([x, y, z])
			pass
		else:
			row, col = self.list.shape
			self.color = self.total / row
			self.total = np.array([0, 0, 0])

	def restartList(self):
		self.list = np.array([[-1, -1, -1]])
