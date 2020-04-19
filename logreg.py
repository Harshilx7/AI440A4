import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from utilities import *



#
# Returns an array containing R,G,B clustered weight distributions by iterating through training data.
# n: cluster layers (must be at least 1)
# data: image data (RGB)
#

def get_rgb(data,n):
    row, col, _ = data.shape
    col = int(col / 2)
    train_data = data[:, :col]
    rgb_arr = np.array([[[0 for i in range(255)] for j in range(2)] for k in range(n)])

    for i in range(row):
        for j in range(col):
            cluster_means = n_cluster((i,j),train_data,n)
            for k in range(n):
                R_i, G_i, B_i = cluster_means[k]
                rgb_arr[k][0][int(R_i)]+=1
                rgb_arr[k][1][int(G_i)]+=1
                rgb_arr[k][2][int(B_i)]+=1

    return rgb_arr



#
# Returns means of different sized clusters, based on number of layers provided
#

def n_cluster(coord,data,layers):
    x,y = coord
    r,g,b = data[x][y]
    if layers == 0:
        return r,g,b
    else:
        x_max, y_max, _ = data.shape
        cluster_means = np.array([])
        prevmean, prevnum = [r,g,b], 1
        for i in range(1,layers+1):
            prevmean, prevnum = cluster_avg(prevmean, prevnum, i, x_max, y_max, coord, data)
            cluster_means.append(prevmean)
            i+=1
    return np.rint(cluster_means)



#
# Returns average of cluster given data about previous layer
#

def cluster_avg(prevmean, prevnum, layer, x_max, y_max, coord, data):
    x,y = coord
    currnum = 0
    prevtot = prevnum*np.array(prevmean)
    tot = np.array([0,0,0])

    for i in range(max(0,x-layer),min(x_max,x+layer)):
        for j in range(max(0,y-layer),min(x_max,y+layer)):
            if(abs(x-i) == layer and abs(y-j) == layer):
                currnum+=1
                tot = np.add(tot, np.array(data[i][j]))
    
    tot = np.add(tot,prevtot)
    means = tot/(prevnum+currnum)
    return means,prevnum+currnum