import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from utilities import *
import random as random
import heapq 

# Returns the sigmoid function output or derivative of the given input
# x: layer array
# deriv: boolean (if True, return sigmoid derivative)
def sigmoid(x,deriv):
    if not deriv:
        return 1/(1+np.exp(-x))
    else:
        return sigmoid(x,False)*sigmoid((1-x),False)

# Returns the softmax function output of the given input, accounting for
# numerical stability by subtracting np.max(x)
# x: layer array
def softmax(x):
    max_i = heapq.nlargest(2,range(len(x)),x.take)
    offset = (x[max_i[0]]+x[max_i[1]])/2
    vals = np.exp(x-offset)
    return vals/vals.sum()

# Returns derivative of the cross entropy loss func. w.r.t. softmax
# y: predicted output array
# y_real: actual output array
def cross_ent_deriv(y, y_real):
    return (y-y_real)/(len(y_real))

# Returns error between the predicted and real output value arrays, using the cross 
# entropy loss function dC/dl_i = -y_real*log(y) 
# y: predicted output array
# y_real: actual output array
def loss(y, y_real):
    n = len(y_real)
    logval = -(y_real*np.log(y))
    loss = np.sum(logval)/n
    return loss

def basicDist(color,color_real):
    x0,y0,z0 = color_real
    x,y,z = color
    return (x-x0)**2+(y-y0)**2+(z-z0)**2

def mapto1(color):
        return (color[0]/255,color[1]/255,color[2]/255)

# Splits the image data into the training and testing data sets.
# Data: image array 
def get_datasets(centroids,data):
    print("Building training dataset")
    centroids = centroids
    row, col, _ = data.shape
    train_clusters = []   
    pix_out = []
    list1 = []
    pix_dict = {}
    for i in range(1,row-1):
        for j in range(1,int(col/2)):
            cluster = getGrayCluster(data,(i, j))
            out_arr = get_pix_out(centroids,data[i][j],pix_dict)
            list1.append((cluster,out_arr)) 

    #randomize training data
    random.shuffle(list1)
    for i in range(len(list1)):
        train_clusters.append(list1[i][0])
        pix_out.append(list1[i][1])
    return train_clusters,pix_out

# Returns list of gray pixels in 3x3 cluster centered on coord
# Data: image data
# coord: coordinate of cluster center  
def getGrayCluster(data,coord):
    x, y = coord
    x_max, y_max, _ = data.shape
    cluster = []
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            cluster.append(gray(tuple(data[i][j]))/255)
    return np.array(cluster)

# Returns a 1 dimensional output array, where each index corresponds to the centroid 
# with the same index in the centroids array. All of the values will be 0 except for
# the entry with the index of the closest centroid to color, whose value will be 1. 
# Centroids: list of centroids
# color: true RGB color of the center pixel of the training patch 
def get_pix_out(centroids,color,pix_dict):
    pix_out = np.zeros(len(centroids))
    minCentroid = ()

    # Find closest centroid
    strColor = np.array2string(color)
    if strColor in pix_dict:
        minCentroid = pix_dict[strColor]

    # Add new color to dictionary if not present 
    else:
        #minDist = calcDistance(centroids[0].color, color)
        minDist = calcDistance(centroids[0],color,True)
        minCentroid = centroids[0]
        for i in range(1, len(centroids)):
            #dist = calcDistance(centroids[i].color, color)
            dist = calcDistance(centroids[i],color,True)
            if dist < minDist:
                minDist = dist
                minCentroid = centroids[i]
        pix_dict[strColor] = minCentroid

    for i in range(len(centroids)):
        if(centroids[i] == minCentroid):
            pix_out[i] = 1
    
    return pix_out

# size: height of the layers
# layers: number of total layers in the NN (including input/output layers; min is 3)
# pix_in: number of input features (generally 9)
# centroids: list of centroids to classify from 
def initializeNN(size,layers,pix_in,centroids):
    w,b = [],[]
    g0_dim,gn_dim = pix_in,len(centroids)

    # Initializes list of weight and bias arrays (first layer is w[0]). The first
    # layer weight array w[0] has dimensions of (size,g0_dim), for example if the 
    # height was 10 nodes and we have 9 initial features the dimension is 10x9. If 
    # we have 8 centroids, the last weight array w[layers-1] will have dimensions 
    # of 8x10. Bias is optional.  

    w.append(np.random.randn(size,g0_dim)) 
    b.append(np.zeros(size))
    for i in range(layers-3):
        w.append(np.random.randn(size,size))
        b.append(np.zeros(size))
    w.append(np.random.randn(gn_dim,size))
    b.append(np.zeros(gn_dim))
    return w,b

# Calculates output with given weights, biases, and input vector.
# w: list of weight arrays
# b: list of bias arrays
# x: input array
def forwardprop(w,b,layers,x):
    x = np.array(x)
    #print(x)
    g = []
    g.append(x) #append g1 (the input)
    #print(g[0])
    if layers>2:
        for i in range(0,layers-2):
            g.append(sigmoid(np.dot(w[i],g[i])+b[i],False))  # append g_2 - g_n-1
            #print(g[i].shape)
    g.append(softmax(np.dot(w[layers-2],g[layers-2])+b[layers-2])) # append g_n
    #print(np.sum(g[layers-1]))
    return g

# Does back propogation to update weights and biases if present. Since we are using
# softmax classification for the last layer, cross entropy is used to compute this
# derivative and the remaining sigmoid function derivatives are computed as usual.  
# g: array containing all layers in the NN for the input cluster
# w: list of weight arrays
# b: list of bias arrays
# y_real: actual output array for the input cluster
# rate: learning rate
def backprop(g,w,b,layers,y_real,rate):
    error = loss(g[len(g)-1],y_real)
    rderivs = [] # Reversed derivatives array (appending from end to start)

    # use cross entropy for last layer's softmax derivative
    d_wn = cross_ent_deriv(g[len(g)-1],y_real) 
    #print(d_wn.shape)
    rderivs.append(d_wn)
    # For remaining layers, we will use the sigmoid derivative
    for i in range(layers-2):
        #print("backprop %i" % i)
        w_i = w[len(w)-1-i] # weight array, going backwards 
        #print(w_i.shape)
        d_zi = np.dot(w_i.T,rderivs[i]) 
        #print(d_zi.shape)
        d_wi = d_zi*sigmoid(g[len(g)-i-2],True)
        rderivs.append(d_wi)
    derivs = rderivs[::-1] # derivatives array

    # Update weights 
    for i in range(0,layers-1):
        #print(w[i].shape)
        #print(g[i].shape)
        #print(derivs[i].shape)
        w[i] -= rate*np.outer(derivs[i],g[i])
        #print(np.outer(derivs[i],g[i]).shape)
        b[i] -= rate*np.sum(derivs[i])
    return w,b,error

# Creates/trains the NN and shows the combined image 
# size: number of nodes per layer
# layers: number of layers
# rate: learning rate
# pix_in: number of input features (generally 9)
# centroids: list of centroids
# data: image data
# iter: number of times to iterate over image data
def trainNN(size,layers,rate,pix_in,centroids,data,iter):
    w,b = initializeNN(size,layers,pix_in,centroids)
    centroiderr = []
    train_data,output = get_datasets(centroids,data)
    print("Training with %i patches" % len(train_data))
    for n in range(iter):
        for i in range(len(train_data)):
            g = forwardprop(w,b,layers,train_data[i])
            w,b,loss = backprop(g,w,b,layers,output[i],rate)
        print("Iteration %i complete" % n)
        curacc = predict_img(w,b,layers,loss,data,centroids)
    print("Finished training")

# Uses NN to make a prediction for the RGB value of the given grayscale cluster's 
# center.
# w: list of arrays
# b: list of bias arrays
# centroids: list of centroid colors
# cluster: input list of 9 pixels in 3x3 cluster
# avg: boolean that says whether or not to take average of 3 highest probability
# centroids
def predict(w,b,layers,centroids,cluster,avg):
    #print(cluster)
    g = forwardprop(w,b,layers,cluster)
    output = g[len(g)-1]
    max = 0
    max_i = heapq.nlargest(3,range(len(output)),output.take)
    colors = [mapto1(centroids[max_i[i]]) for i in range(3)]

    #predict using 3 highest probability centroids
    if avg:
        diff1 = output[max_i[0]]-output[max_i[1]]
        diff2 = output[max_i[0]]-output[max_i[2]]
        if(diff1<.01 and diff2<.01):
            R = sum(centroids[i][0] for i in max_i)/3
            G = sum(centroids[i][1] for i in max_i)/3
            B = sum(centroids[i][2] for i in max_i)/3
            return mapto1((R,G,B))
        elif(diff1<.01):
            R = (centroids[max_i[0]][0] + centroids[max_i[1]][0])/2
            G = (centroids[max_i[0]][1] + centroids[max_i[1]][1])/2
            B = (centroids[max_i[0]][2] + centroids[max_i[1]][2])/2
            return mapto1((R,G,B))
        else:
            return mapto1(centroids[max_i[0]])
    #predict using highest probability centroid
    else:
        return mapto1(centroids[max_i[0]])
    
# Uses NN to make predictions for the right half of the image given, and shows the 
# corresponding combined image
# Loss: Cross Entropy loss calcualted from backprop
def predict_img(w,b,layers,loss,data,centroids):
    print("Starting predictions")
    row, col, _ = data.shape
    newimg = []
    diffs = []
    pix_dict = {}
    z = 0
    for i in range(1,row-1):
        imgrow = []
        for j in range(1,col-1):
            if(j<int(col/2)):
                imgrow.append(mapto1(data[i][j]))
            else:
                cluster = getGrayCluster(data,(i, j))
                center_pix = cluster[4]
                color = predict(w,b,layers,centroids,cluster,False)
                realcolor = mapto1(data[i][j])
                #out_arr = get_pix_out(centroids,data[i][j],pix_dict)
                #realcentroid = mapto1(centroids[np.argmax(out_arr)])
                error = basicDist(color,realcolor)
                #error = basicDist(color,realcentroid)
                imgrow.append(color)
                diffs.append(error)
        newimg.append(imgrow)
    print("Predictions complete")
    acc = sum(diffs)/len(diffs)
    print("Acc: %.5f %%" % (100*(1-acc)))
    #if(100*(1-acc)>95):    
    plt.imshow(np.array(newimg))
    plt.title("Centroids: %i Loss: %.5f Accuracy: %.5f %%" % 
    (len(centroids),loss,100*(1-acc)))
    plt.show()
    return 100*(1-acc)










        

        



        



