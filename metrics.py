# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:27:58 2022

@author: Acer
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class Metrics:
    def __init__(self):
        pass
    
    @staticmethod
    def metrics(X):
        Sum_of_squared_distances = []
        K = range(2,10)
        for k in K:
           km = KMeans(n_clusters=k, max_iter=200, n_init=10)
           km = km.fit(X)
           Sum_of_squared_distances.append(km.inertia_)
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        return plt.show()