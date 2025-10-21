import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise  import cosine_similarity
from nltk.stem import WordNetLemmatizer
import pandas as pd
import warnings
import streamlit as st
import numpy as np
warnings.filterwarnings('ignore')
#import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

import librosa
librosa.show_versions()

import streamlit as st
import speech_recognition as sr
import pyttsx3
import time
from datetime import datetime

#download required NLTK data
#nltk.download('stopwords')
#nltk.download('punk')
#nltk.download('wordnet')
#nltk.download('punkt_tab')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header=None)
data

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

sales = sales[1].reset_index(drop = True)
cust = cust[1].reset_index(drop = True)

new_data = pd.DataFrame()
new_data['Question'] = cust
new_data['Answer'] = sales

new_data

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric
        # The code above does the following:
        # Identifies every word in the sentence
        # Turns it to a lower case
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return ' '.join(preprocessed_sentences)


new_data['tokenized Questions'] = new_data['Question'].apply(preprocess_text)
new_data


xtrain = new_data['tokenized Questions'].to_list()
xtrain
