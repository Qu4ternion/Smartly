# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:52:59 2022

@author: Acer
"""


from sklearn.cluster import KMeans
import json

class kmeans:
    def __init__(self):
        pass
    
    def cluster(X, stems):
        
        km = KMeans(n_clusters=2, max_iter=200, n_init=10)
        km = km.fit(X)
        
        
        # Predicted labels:
        preds = km.labels_
        
        # Labelling the preds:
        labels = list(zip(preds, stems))
        
        
        # Organize into K different groups in a dict:
        result = {}
        
        for label in labels:
            if label[0] in result.keys():
                result[label[0]].append(label[1])
            
            else:
                result[ int(label[0]) ] = [label[1]]
        
        
        
        # Format in a JSON:
        return json.dumps(result, indent=2)