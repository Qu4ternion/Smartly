# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:26:39 2022

@author: Acer
"""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessor:
    
    def __init__(self):
        pass
    
    def preprocess(self, data):
        
        # Tokenization:
        tokens = []
        for sentence in data:    
           for word in word_tokenize(sentence):
               tokens.append(word)
        
        
        # Lower-casing:
        lowered = [word.lower() for word in tokens]


        # Remove stop-words:
        stops = [word for word in lowered if word not in stopwords.words('french')]
        
        
        # Remove non alphanumeric characters:
        alphanum = [word for word in stops if word.isalnum()]
        
        
        # Stemming:
        porter = PorterStemmer()
        stems = [ porter.stem(word) for word in alphanum]
        
        return stems