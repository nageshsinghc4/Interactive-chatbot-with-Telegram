#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:53:11 2019

@author: nageshsinghchauhan
"""


import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim
from sklearn.metrics.pairwise import pairwise_distances_argmin
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
dialogues = pd.read_csv("/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/dialogues.tsv",sep="\t")
posts = pd.read_csv("/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/tagged_posts.tsv",sep="\t")

#Create training data for intent classifier - Chitchat/SO Question
texts  =  list(dialogues[:200000].text.values) + list(posts[:200000].title.values)
labels =  ['dialogue']*200000 + ['stackoverflow']*200000

data = pd.DataFrame({'text':texts,'target':labels})


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    return text.strip()

# Doing some data cleaning
data['text'] = data['text'].apply(lambda x : text_prepare(x))
X_train, X_test, y_train, y_test = train_test_split(data['text'],data['target'],test_size = .1 , random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

#Create Intent classifier
# We will keep our models and vectorizers in this folder
def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""
    tfv = TfidfVectorizer(dtype=np.float32, min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    X_train = tfv.fit_transform(X_train)
    X_test = tfv.transform(X_test)
    
    pickle.dump(tfv,vectorizer_path)
    return X_train, X_test

X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, open("/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/tfidf.pkl",'wb'))
intent_recognizer = MultinomialNB()
intent_recognizer.fit(X_train_tfidf,y_train)

# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(intent_recognizer, open("/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/intent_clf.pkl" , 'wb'))

# Create Programming Language classifier¶
X = posts['title'].values
y = posts['tag'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open("/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/tfidf.pkl", 'rb'))
X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)

tag_classifier = OneVsRestClassifier(LogisticRegression(C=5,random_state=0))
tag_classifier.fit(X_train_tfidf,y_train)

# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(tag_classifier, open("/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/tag_clf.pkl", 'wb'))

#Store Question database Embeddings
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/GoogleNews-vectors-negative300.bin', binary=True)

#We want to convert every question to an embedding and store them. 
#Whenever user asks a stack overflow question we want to use cosine similarity to get the most similar question
def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    word_tokens = question.split(" ")
    question_len = len(word_tokens)
    question_mat = np.zeros((question_len,dim), dtype = np.float32)
    
    for idx, word in enumerate(word_tokens):
        if word in embeddings:
            question_mat[idx,:] = embeddings[word]
            
    # remove zero-rows which stand for OOV words       
    question_mat = question_mat[~np.all(question_mat == 0, axis = 1)]
    
    # Compute the mean of each word along the sentence
    if question_mat.shape[0] > 0:
        vec = np.array(np.mean(question_mat, axis = 0), dtype = np.float32).reshape((1,dim))
    else:
        vec = np.zeros((1,dim), dtype = np.float32)
    return vec

counts_by_tag = posts.groupby(by=['tag'])["tag"].count().reset_index(name = 'count').sort_values(['count'], ascending = False)
counts_by_tag = list(zip(counts_by_tag['tag'],counts_by_tag['count']))
print(counts_by_tag)

for tag, count in counts_by_tag:
    tag_posts = posts[posts['tag'] == tag]
    tag_post_ids = tag_posts['post_id'].values
    tag_vectors = np.zeros((count, 300), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = question_to_vec(title, model, 300)
    # Dump post ids and vectors to a file.
    filename = '/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/embeddings_folder/'+ tag + '.pkl'
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))
    
#Given a question and tag can I retrieve the most similar question post_id
def get_similar_question(question,tag):
    # get the path where all question embeddings are kept and load the post_ids and post_embeddings
    embeddings_path = '/Users/nageshsinghchauhan/Documents/projects/chatbot/stackoverflow/data/embeddings_folder/' + tag + ".pkl"
    post_ids, post_embeddings = pickle.load(open(embeddings_path, 'rb'))
    # Get the embeddings for the question
    question_vec = question_to_vec(question, model, 300)
    # find index of most similar post
    best_post_index = pairwise_distances_argmin(question_vec,
                                                post_embeddings)
    # return best post id
    return post_ids[best_post_index]

get_similar_question("how to use list comprehension in python?",'python')



