#!/usr/bin/env python
# coding: utf-8

# # Singular Value Decomposition (SVD) & Its Application In Recommender System

# Singular Value Decomposition (SVD), a classical method from linear algebra is getting popular in the field of data science and machine learning. This popularity is because of its application in developing recommender systems. There are a lot of online user-centric applications such as video players, music players, e-commerce applications, etc., where users are recommended with further items to engage with.

# Finding and recommending many suitable items that would be liked and selected by users is always a challenge. There are many techniques used for this task and SVD is one of those techniques. This article presents a brief introduction to recommender systems, an introduction to singular value decomposition and its implementation in movie recommendation. 

# **What is a Recommender System?**
# A recommender system is an intelligent system that predicts the rating and preferences of users on products. The primary application of recommender systems is finding a relationship between user and products in order to maximise the user-product engagement. The major application of recommender systems is in suggesting related video or music for generating a playlist for the user when they are engaged with a related item.

# **Singular Value Decomposition**
# The Singular Value Decomposition (SVD), a method from linear algebra that has been generally used as a dimensionality reduction technique in machine learning. SVD is a matrix factorisation technique, which reduces the number of features of a dataset by reducing the space dimension from N-dimension to K-dimension (where K<N). In the context of the recommender system, the SVD is used as a collaborative filtering technique. It uses a matrix structure where each row represents a user, and each column represents an item. The elements of this matrix are the ratings that are given to items by users.

# **Singular Value Decomposition (SVD) based Movie Recommendation**
# Below is an implementation of singular value decomposition (SVD) based on collaborative filtering in the task of movie recommendation.
# This task is implemented in Python. For simplicity,
# the MovieLens 1M Dataset has been used.
# This dataset has been chosen because it does not require any preprocessing as the main focus of this article is on SVD and recommender systems.

# **Import the required python libraries:**

# In[1]:


import pandas as pd
import numpy as np


# **Read the dataset from where it is downloaded in the system. It consists of two files ‘ratings.dat’ and ‘movies.dat’ which need to be read.**

# In[2]:


data1 = pd.read_csv('ratings.txt')
data1.head()


# **here we haven't any column name so we have name our columns**

# In[3]:


data1.shape


# **If we read dataset simplly from traditional method so it retrives data as like in dataset so we have clean than using some useful arguments or pandas library**

# **for naming of column names and eleminate our delimiter from data set**

# In[4]:


data = pd.io.parsers.read_csv('ratings.txt', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
data.head(10)


# In[5]:


movie_data = pd.io.parsers.read_csv('movies1.txt',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')
movie_data.head(10)


# In[6]:


movie_data.isnull().sum()


# In[7]:


data.describe


# In[8]:


movie_data.info


# **Create the rating matrix with rows as movies and columns as users.**

# In[9]:


ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values


# In[10]:


ratings_mat


# **Normalise the matrix.**

# In[11]:


normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T


# In[12]:


normalised_mat


# **Compute the Singular Value Decomposition (SVD).**

# In[13]:


A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)


# In[14]:


A


# **Define a function to calculate the cosine similarity. Sort by most similar and return the top N results.**

# In[15]:


def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1197 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]


# **If change movie_id we will receive defferent recommended/Recommendation movies**

# **Define a function to print top N similar movies.**

# In[16]:


def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])


# **Initialise the value of k principal components, id of the movie as given in the dataset, and number of top elements to be printed.**

# In[17]:


k = 50
movie_id = 10 # (getting an id from movies.txt)
top_n = 10 # how many movies you want
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)


# **Print the top N similar movies.**

# In[18]:


print_similar_movies(movie_data, 20, indexes)


# In[19]:


print_similar_movies(movie_data, movie_id, indexes)


# In[32]:


print_similar_movies(movie_data, 13, indexes)


# **Binding all together:**
# 
# **Importing Libraries**

# ## This method sir has given me above metion method I have made my self using this method

# In[21]:


import numpy as np
import pandas as pd


# **Reading dataset (MovieLens 1M movie ratings dataset: downloaded from** https://grouplens.org/datasets/movielens/1m/)

# In[22]:


data = pd.io.parsers.read_csv('ratings.txt', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('movies1.txt',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')


# In[23]:


data.isnull().sum()


# In[24]:


movie_data


# **Creating the rating matrix (rows as movies, columns as users)**

# In[25]:


ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values


# **Normalizing the matrix(subtract mean off)**

# In[26]:


normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T


# **Computing the Singular Value Decomposition (SVD)**

# In[27]:


A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)


# **Function to calculate the cosine similarity (sorting by most similar and returning the top N)**

# In[28]:


def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]


# # Function to print top N similar movies

# In[29]:


def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])


# **k-principal components to represent movies, movie_id to find recommendations, top_n print n results**

# In[30]:


k = 50
movie_id = 10 # (getting an id from movies.dat)
top_n = 10
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)


# ## Printing the top N similar movies

# In[31]:


print_similar_movies(movie_data, movie_id, indexes)


# **A similar application is in the field of e-commerce where customers are recommended with the related products, but this application involves some other techniques such as association rule learning. It is also used to recommend contents based on user behaviours on social media platforms and news websites.**
# 

# There are two popular approaches used in recommender systems to suggest items to the users:-
# 
# ***Collaborative Filtering:*** **The assumption of this approach is that people who have liked an item in the past will also like the same in future. This approach builds a model based on the past behaviour of users. The user behaviour may include previously watched videos, purchased items, given ratings on items. In this way, the model finds an association between the users and the items. The model is then used to predict the item or a rating for the item in which the user may be interested. Singular value decomposition is used as a collaborative filtering approach in recommender systems.**

# ***Content-Based Filtering:*** **This approach is based on a description of the item and a record of the user’s preferences. It employs a sequence of discrete, pre-tagged characteristics of an item in order to recommend additional items with similar properties. This approach is best suited when there is sufficient information available on the items but not on the users. Content-based recommender systems also include the opinion-based recommender system.
# Apart from the above two approaches, there are few more approaches to build recommender systems such as multi-criteria recommender systems, risk-aware recommender systems, mobile recommender systems, and hybrid recommender systems (combining collaborative filtering and content-based filtering).**
