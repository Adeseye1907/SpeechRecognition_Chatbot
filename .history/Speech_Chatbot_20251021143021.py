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


#vectorise corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)
print(corpus)


bot_greeting = ['Hello User! Do you have any questions?',
                'Hey  you! tell me what you want',
                'I am like a genie in a bottle. Hit me with your question.',
                'Hi! how can i help you today?']

bot_farewell= ['Thanks for your usage... bye.',
              'I hope you had a good experience.',
              'Have a great day and keep enjoying Samsung.']


human_greeting= ['hi','hello', 'good day', 'hey', 'hola' ]

human_exit = ['thank you', 'thanks', 'bye bye', 'goodbye', 'quit']


import random
random_greeting = random.choice(bot_greeting)
random_farewell = random.choice(bot_farewell)

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>ORGANISATIONAL SPEECH CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Adeseye</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.header('Project Background Information', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ. ")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


col1, col2, = st.columns(2)
col2.image('pngwing.com (3).png')

userPrompt = st.chat_input('Ask Your Question')
if userPrompt:
    col1.chat_message("human").write(userPrompt)

    userPrompt = userPrompt.lower()
    if userPrompt in human_greeting:
        col1.chat_message("ai").write(random_greeting)
    elif userPrompt in human_exit:
        col1.chat_message("ai").write(random_farewell)
    else:
        proUserInput = preprocess_text(userPrompt)
        vect_user = tfidf_vectorizer.transform([proUserInput])
        similarity_scores = cosine_similarity(vect_user, corpus)
        most_similar_index = np.argmax(similarity_scores)
        col1.chat_message("ai").write(new_data['Answer'].iloc[most_similar_index])
