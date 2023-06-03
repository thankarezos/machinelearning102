from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import tensorflow as tf
import pandas as pd
import plots as pl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

pca = PCA(n_components=2)



teamsOriginal = np.load("data.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)
labels -= 1

teams = teamsOriginal[ : , : , 1: ]

for i in range(0, 30):
    x = pca.fit_transform(teams[i])
    kmeans = KMeans(n_clusters=6, random_state=0, n_init=1).fit(x)
    plt.cla()
    plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1] , marker='s', s=500, alpha=0.5)
    # plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1] , 'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    plt.pause(3)
    

