import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from utilities import *

# Returns the sigmoid function output of the given input
def sigmoid(x,deriv):
    if not deriv:
        return 1/(1+np.exp(-x))
    else:
        return x*(1-x)

# Returns the softmax function output of the given input
def softmax(x):
    vals = np.exp(x)
    return vals/vals.sum

# Returns derivative of the cross entropy loss func. w.r.t. softmax
# y: predicted output array
# y_real: actual output array
def cross_ent_deriv(y, y_real):
    return (y-y_real)/(y_real.shape[0])

# Returns error between the predicted and real output value arrays, using the cost 
# function -y_real*log(y)
# y: predicted output array
# y_real: actual output array
def loss(y, y_real):
    n = len(y_real)
    logp = -1*np.log(y[np.arange(n), y_real.argmax(axis=1)])
    loss = np.sum(logp)/n
    return loss

class NN:
    input_pix = 9
    data = []
    centroids = []
    pix_dict = {}

    # size: height of the layers
    # layers: number of total layers in the NN (including input/output layers; min is 3)
    # rate: learning rate for back propogation
    # pix_in: number of input features (generally 9)
    # centroids: list of centroids to classify from 
    # data: image data
    def __init__(self,size,layers,rate,pix_in,centroids,data):
        # initialize values based on inputs
        self.data = data
        self.rate = rate
        self.g = []
        g0_dim,gn_dim = pix_in,len(centroids)
        
        # Initializes list of weight and bias arrays (first layer is w[0]). The first
        # layer weight array w[0] has dimensions of (size,g0_dim), for example if the 
        # height was 10 nodes and we have 9 initial features the dimension is 10x9. If 
        # we have 8 centroids, the last weight array w[layers-1] will have dimensions 
        # of 8x10. Bias is optional.  

        self.w = []
        self.b = []
        self.w.append(np.random.randn(size,g0_dim))
        #self.b.append(np.zeroes(size,1))
        for i in range(layers-3):
            self.w.append(np.random.randn(size,size))
            self.b.append(np.zeroes(size,1))
        w.append(np.random.randn(gn_dim,size))
        b.append(np.zeroes(gn_dim,1))

    # Returns a 1 dimensional output array, where each index corresponds to the centroid 
    # with the same index in the centroids array. All of the values will be 0 except for
    # the entry with the index of the closest centroid to color, whose value will be 1. 
    # Centroids: list of centroids
    # color: true RGB color of the center pixel of the training patch 
    def get_pix_out(self,centroids,color):
        pix_out = np.zeroes(len(centroids))
        minCentroid = ()

        # Find closest centroid
        strColor = np.array2string(color)
        if strColor in self.pix_dict:
            minCentroid = self.pix_dict[strColor]

        # Add new color to dictionary if not present 
        else:
            minDist = calcDistance(centroids[0].color, color)
            minCentroid = centroids[0]
            for i in range(1, len(centroids)):
                dist = calcDistance(centroids[i].color, color)
                if dist < minDist:
                    minDist = dist
                    minCentroid = centroids[i]
            dictionary[strColor] = minCentroid

        for i in len(centroids):
            if(self.centroids[i] == minCentroid):
                pix_out[i] = 1
        
        return pix_out

    # Does the forward calculation of the layers using current weights. Returns an 
    # array containing node values for each layer, g[i]. 
    # x: list of grayscale pixels in the patch
    def forwardprop(self,x):
        layers = self.layers
        g = []
        g.append(sigmoid(np.dot(self.w[0],x))) # append g_1
        for i in range(layers-3):
            g.append(sigmoid(np.dot(self.w[i+1],g[i]))) # append g_2 - g_n-1
        g.append(softmax(np.dot(self.w[layers-1],g[layers-2]))) # append g_n
        return g

    # Does back propogation to update weights and biases if present. Since we are using
    # softmax classification for the last layer, cross entropy is used to compute this
    # derivative and the remaining sigmoid function derivatives are computed as usual.  
    # g: array containing all layers in the NN for the input cluster
    # p: actual output array for the input cluster
    def backprop(self,g,p):
        error = loss(g[len(g)-1],p)
        print("Loss: ",error)
        dgn = cross_ent_deriv(g[len(g)-1],p)
        #use cross entropy for last layer's softmax derivative

        #use sigmoid derivatives for remaining layers

    # Trains the NN and shows the combined image 
    def train_NN(self):
        train_data,output = get_datasets(self)
        for i in len(train_data):
            g = forwardprop(self,train_data[i])
            backprop(self,g,output[i])

    # Splits the image data into the training and testing data sets. 
    def get_datasets(self):
        centroids = self.centroids
        row, col, _ = self.data.shape
        train_clusters = []   
        test_clusters = []
        pix_out = []

        for i in range(1,row-1):
            for j in range(1,col/2):
                cluster = getGrayCluster((i, j))
                train_clusters.append(cluster)
                pix_out.append(get_pix_out(centroids,data[i][j]))

        return train_clusters,pix_out,test_clusters

    # Returns list of gray pixels in 3x3 cluster centered on coord
    def getGrayCluster(self,coord):
        data = self.data
        x, y = coord
        x_max, y_max, _ = data.shape
        cluster = []
        for i in range(max(0,x-1),min(x+1,x_max)):
            for j in range(max(0,y-1),min(y+1,y_max)):
                cluster.append(gray(tuple(data[i][j])))
        return np.array(cluster)

    # Uses NN to make predictions for the right half of the image given, and shows the 
    # corresponding combined image
    def predict_img(self):
        data = self.data
        row, col, _ = data.shape
        newimg = []

        for i in range(1,row-1):
            imgrow = []
            for j in range(1,col-1):
                if(j<col/2):
                    imgrow.append(data[i][j])
                else:
                    cluster = getGrayCluster((i, j), data)
                    color = predict(cluster)
                    imgrow.append(color)
            newimg.append(imgrow)

        plt.imshow(newimg)
        return newimg

    # Uses NN to make a prediction for the RGB value of the given grayscale cluster's 
    # center. 
    def predict(self,cluster):
        g = forwardprop(cluster)
        output = g[len(g)-1]
        max_i = np.argmax(output,axis = 0)
        return self.centroids[max_i]







        

        



        



