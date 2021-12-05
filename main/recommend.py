
from ctypes import sizeof
import time
from types import new_class
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from matplotlib import pyplot
from gensim.models import KeyedVectors

import sys

import pickle


def get_model():
    return Word2Vec.load(R"main/finetuned_model.model")



def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def get_and_clean_data():
    df = pd.read_csv(R"main/data.csv")
    
    df['Desc'] = df['Desc'].astype(str)
    df['cleaned'] = df['Desc'].apply(_removeNonAscii)

    df['cleaned'] = df.cleaned.apply(func = make_lower_case)
    df['cleaned'] = df.cleaned.apply(func = remove_stop_words)
    df['cleaned'] = df.cleaned.apply(func=remove_punctuation)
    df['cleaned'] = df.cleaned.apply(func=remove_html)

    return df

def vectors(df,model):
    word_embeddings = []


    

    for line in df['cleaned']:
        avgword2vec = None
        count = 0
        for word in line.split():
            if word in model.wv.index_to_key:
                count += 1
                if avgword2vec is None:
                    avgword2vec = model.wv.get_vector(word)
                else:
                    avgword2vec = avgword2vec + model.wv.get_vector(word)
                
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
        
            word_embeddings.append(avgword2vec)
            


    pickle.dump( word_embeddings, open( R"C:\Users\Jonathan\Documents\vit\vit\useless docs\sem7\sain\project\w2v\recui\main\word_embeddings", "wb" ) )



def recommendations(title,df,model):
    

    embeddings = pickle.load( open( R"C:\Users\Jonathan\Documents\vit\vit\useless docs\sem7\sain\project\w2v\recui\main\word_embeddings", "rb" ) )
    now = time.time()

    cosine_similarities = cosine_similarity(embeddings, embeddings)

    books = df[['title', 'image_link']]
    indices = pd.Series(df.index, index = df['title']).drop_duplicates()
         
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    recommend = books.iloc[book_indices]

    titles = []
    for index, row in recommend.iterrows():

        titles.append(row['title'])
    
    return titles

def search(keyword, df):
    search = '|'.join(keyword)
    searched = df[df['text'].str.contains(search, na=False)]
    return searched

df = get_and_clean_data()


model = get_model()