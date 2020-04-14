import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np


#
# converts image pixel to grayscale
#
# rgb: rgb values of the pixel
#
def gray(rgb):
	r, g, b = rgb
	n = int((0.21 * r) + (0.72 * g) + (0.07 * b))
	return [n] * 3


def main():
	data = image.imread('test_image.jpg')
	data = np.array(data)

	# makes the colored picture into grayscale
	newData = [[gray(y) for y in x]for x in data]
	newData = np.array(newData)

	plt.imshow(newData)
	plt.show()
	pass


if __name__ == '__main__':
	main()
