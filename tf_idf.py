# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:49:43 2022

@author: Acer
"""

from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:
    def __init__(self):
        pass
    
    def Vectorize(self, stems):
        tf_idf = TfidfVectorizer()
        X = tf_idf.fit_transform(stems)
        
        return X