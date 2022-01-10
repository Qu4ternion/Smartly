# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:08:58 2022

@author: Acer
"""

import streamlit as st
from preprocessing import Preprocessor
from tf_idf import TFIDF
from model import kmeans
from metrics import Metrics
from io import StringIO 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Add title:
st.title('Smartly API:')

# Add a subheader:
st.subheader('Clustering de phrases inconnues')

st.write('Téléchargez votre fichier texte ci-dessous:')

# Add file upload:
file = st.file_uploader('Fichiers à télécharger:')


# Uploaded file handling:
if file is not None:

    # Downloaded file:
    data = StringIO(file.getvalue().decode("latin1"))
    
    # Pre-processing data:
    cleaned = Preprocessor.preprocess(data, data)
    
    # Vectorisation TF-IDF:
    features = TFIDF.Vectorize(cleaned, cleaned)
    
    # Running model:
    clustering = kmeans.cluster(features, cleaned)
    
    # Result:
    st.subheader('API JSON response:')
    st.write(clustering)

    
    # Metrics 
    st.subheader('Metrics:')
    
    Sum_of_squared_distances = []
    K = range(2,10)
    for k in K:
       km = KMeans(n_clusters=k, max_iter=200, n_init=10)
       km = km.fit(features)
       Sum_of_squared_distances.append(km.inertia_)
    
    # Graph:
    fig, ax = plt.subplots()
    ax.plot(K, Sum_of_squared_distances, 'bx-')
    ax.set_xlabel('Groupes K')
    ax.set_ylabel('Sommes des carrés')
    ax.set_title('Méthode du coude pour nombre optimal de groupes')
    st.pyplot(fig)