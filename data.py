# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:50:55 2022

@author: Acer
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import json


# Data:
path = r'C:\Users\Acer\Desktop\Smartly\conversations.yml'

with open(path, 'r') as f:
    file = f.read()


# Split breaklines:
data = file.split('\n')


# Remove left marks:
for i in range(len(data)):
    data[i] = data[i].lstrip('- ')
    


# Write cleaned data:
#write_path = r'C:\Users\Acer\Desktop\Smartly'
#with open(path, 'w') as f:
#    file = f.write(data)


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



# Vectorization:
tf_idf = TfidfVectorizer()
X = tf_idf.fit_transform(stems)
   

# K-means clustering:
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
print(json.dumps(result, indent=2))