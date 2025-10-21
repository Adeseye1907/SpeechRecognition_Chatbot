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


#download required NLTK data
#nltk.download('stopwords')
#nltk.download('punk')
#nltk.download('wordnet')
#nltk.download('punkt_tab')