### This script allows you to generate your own exam questions and answers

### First import relevant libraries

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import scipy
from scipy.sparse import csr_matrix
import nltk
from sklearn.metrics.pairwise import pairwise_distances
from numpy import linalg as LA
from sklearn.decomposition import PCA



### Question 1

### Input data with original values

data_1 = {'Weight': [1, 2, 4, 5], 'pH-Index': [1, 1 , 3, 4]}
data_1 = pd.DataFrame(data=data_1)
data_1 = data_1.rename({0: 'A', 1: 'B', 2: 'C', 3: 'D'}, axis='index')

data_1



### Euclidean Distance Kmeans

initial_points = [[1,1],[5,5]]

kmeans = KMeans(n_clusters=2, init=np.array(initial_points), n_init=1, max_iter=300, tol=0.0001, 
       precompute_distances=True, verbose=1, random_state=None, copy_x=True, n_jobs=1, algorithm="auto")

kmeans.fit(data_1)

print("")
print("Cluster Allocations")
print(kmeans.labels_)

print("")
print("Centroids")
print(kmeans.cluster_centers_)

### Let's do PCA

data_2 = {'x': [2.3, 2, 1, 1.5, 1.1], 'y': [2.7, 1.6, 1.1, 1.6, 0.9]}
data_2 = pd.DataFrame(data=data_2)

print("Means")
print(data_2.mean())

print("")

RowZeroMeanData = pd.DataFrame(data_2 - data_2.mean())

print("Zero Mean Data")
print(RowZeroMeanData)
print("")

c = RowZeroMeanData.cov()

print(c)

eigenvalues, eigenvectors = LA.eig(c.values)

print("")
print("Eigenvalues")
print(eigenvalues)
print(" ")
print("Eigenvectors")
print(eigenvectors)

print(" ")
pca = PCA(n_components=1)
print("One-Dimensional Data for First Principle Component")
print(pca.fit_transform(data_2))



### Let's do SVM 

from sklearn import svm
import matplotlib.pyplot as plt

data_3 = {'x': [7, 4, 7, 4.5, 5], 'y': [3, 7, 6, 5, 4],'Class': [0, 0, 1, 1, 1]}
data_3 = pd.DataFrame(data=data_3)

data_3["|x-y|"] = abs(data_3["x"]-data_3["y"])

print("Please list new dataset")
print(data_3)

print("")
print("Find the support vectors in 3D space")
plt.scatter(data_3[["Class"]].values,data_3[["|x-y|"]].values,s=15)
plt.show()

print("")
print("Calculate margin of hyperplane")

class_means = data_3[['Class','|x-y|']].groupby("Class").mean().values

min_0 = data_3.loc[data_3['Class'] == 0]

min_0 = min_0[["|x-y|"]].min().values[0]

max_0 = data_3.loc[data_3['Class'] == 0]

max_0 = max_0[["|x-y|"]].max().values[0]

min_1 = data_3.loc[data_3['Class'] == 1]

min_1 = min_1[["|x-y|"]].min().values[0]

max_1 = data_3.loc[data_3['Class'] == 1]

max_1 = max_1[["|x-y|"]].max().values[0]

print("")
print("Support Vectors are data points below")

if(class_means[0] > class_means[1]):
    print(data_3[(data_3["|x-y|"]== min_0) | (data_3["|x-y|"]== max_1)])
    support_vectors = data_3[(data_3["|x-y|"]== min_0) | (data_3["|x-y|"]== max_1)]
    print("")
    print("Hyperplane margin is ")
    print((support_vectors[["|x-y|"]].max()-support_vectors[["|x-y|"]].min()).values[0])
    
elif(class_means[0] < class_means[1]):
    print(data_3[(data_3["|x-y|"]== min_1) | (data_3["|x-y|"]== max_0)])
    support_vectors = data_3[(data_3["|x-y|"]== min_1) | (data_3["|x-y|"]== max_0)]
    print("")
    print("Hyperplane margin is ")
    print((support_vectors[["|x-y|"]].max()-support_vectors[["|x-y|"]].min()).values[0])