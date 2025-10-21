# --- Import Required Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import random
import time
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- Download NLTK Resources ---
nltk.download('punkt')
nltk.download('wordnet')

# --- Initialize Lemmatizer ---
lemmatizer = WordNetLemmatizer()

# --- Load and Prepare Data ---
data = pd.read_csv('Samsung Dialog.txt', sep=':', header=None, names=['Speaker', 'Line'])
cust = data.loc[data['Speaker'] == 'Customer', 'Line'].reset_index(drop=True)
sales = data.loc[data['Speaker'] == 'Sales Agent', 'Line'].reset_index(drop=True)

# Make sure lengths match
min_len = min(len(cust), len(sales))
cust, sales = cust[:min_len], sales[:min_len]

new_data = pd.DataFrame({'Question': cust, 'Answer': sales})

# --- Define Text Preprocessing Function ---
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    return ' '.join(preprocessed_sentences)

new_data['tokenized'] = new_data['Question'].apply(preprocess_text)

# --- Vectorize Corpus ---
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(new_data['tokenized'])

# --- Greeting Lists ---
bot_greeting = [
    'Hello User! Do you have any questions?',
    'Hey you! Tell me what you want.',
    'I am like a genie in a bottle. Hit me with your question.',
    'Hi! How can I help you today?'
]

bot_farewell = [
    'Thanks for your usage... bye.',
    'I hope you had a good experience.',
    'Have a great day and keep enjoying Samsung.'
]

human_greeting = ['hi', 'hello', 'good day', 'hey', 'hola']
human_exit = ['thank you', 'thanks', 'bye', 'goodbye', 'quit']

# --- Define Speech Transcription Function ---
def transcribe_speech(language_code="en-US"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now... (Keep it short)")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        st.info("📝 Transcribing your voice...")
        try:
            text = r.recognize_google(audio, language=language_code)
            st.success(f"✅ Transcription complete: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand your voice. Please try again.")
            return None
        except sr.RequestError:
            st.error("⚠️ Could not reach the speech recognition service.")
            return None

# --- Streamlit UI ---
st.markdown("<h1 style='text-align:center; color:#0C2D57;'>🎤 ORGANIZATIONAL SPEECH CHATBOT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#F11A7B;'>Built by Adeseye</h4>", unsafe_allow_html=True)

st.markdown("---")
st.write("This chatbot can respond to your text or speech questions based on a company FAQ file. "
         "You can either type your question or click the button to speak.")

input_type = st.radio("Choose your input method:", ["Text", "Speech"])

user_input = None

if input_type == "Text":
    user_input = st.chat_input("Ask your question here...")
elif input_type == "Speech":
    if st.button("🎙️ Click to Record Speech"):
        user_input = transcribe_speech()

if user_input:
    st.chat_message("human").write(user_input.lower())
    user_input = user_input.lower()

    if user_input in human_greeting:
        response = random.choice(bot_greeting)
    elif user_input in human_exit:
        response = random.choice(bot_farewell)
    else:
        processed_input = preprocess_text(user_input)
        vect_input = tfidf_vectorizer.transform([processed_input])
        similarity_scores = cosine_similarity(vect_input, corpus)
        most_similar_index = np.argmax(similarity_scores)
        response = new_data['Answer'].iloc[most_similar_index]

    st.chat_message("ai").write(response)
