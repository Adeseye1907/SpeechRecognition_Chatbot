# --- Import Required Libraries ---
import streamlit as st
import pandas as pd
import numpy as np # <-- MISSING
import nltk
import os
import random # <-- MISSING
import time # <-- MISSING
import ssl # <-- MISSING (Needed for the fix below)
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer # <-- MISSING
from sklearn.metrics.pairwise import cosine_similarity # <-- MISSING
from nltk.tokenize import word_tokenize # This is fine
from nltk.corpus import stopwords # This is fine
from nltk.stem import WordNetLemmatizer # This is fine
# from nltk.tokenize import sent_tokenize # Optional: You can import this too

# ============================
# 1. Ensure Required NLTK Data is Available
# ============================
nltk_packages = [
    "punkt",
    "stopwords",
    "wordnet"
]
# --- NLTK Imports (Ensure 'os' and 'ssl' are imported at the very top) ---
import nltk
import ssl
import os 
# ... other imports ...

# ========================================================================
# 1. FINAL FIX: Robust NLTK Data Download and Path Configuration
# ========================================================================

# 1. Set a consistent, writable NLTK data path (often fixes cloud path issues)
# This path is usually writable on Streamlit Cloud
nltk_data_path = "/tmp/nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)
os.makedirs(nltk_data_path, exist_ok=True)


# 2. SSL Fix (Keep this, it's necessary for the download process itself)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# 3. Force Downloads to the Writable Path
try:
    # Use download_dir to force installation to the correct path
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True, force=True) 
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True, force=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True, force=True) 
    nltk.download('omw-1.4', download_dir=nltk_data_path, quiet=True, force=True)

    # CRITICAL: Verify the resource is found in the path we set
    nltk.data.find('tokenizers/punkt')

except Exception as e:
    st.error(f"🚨 NLTK Download Failed: Cannot locate essential data. Error: {e}")
    st.stop()

# --- Initialize Lemmatizer ---
lemmatizer = WordNetLemmatizer()
# --- Load and Prepare Data ---
# Ensure 'Samsung Dialog.txt' is in the same directory as this script.
try:
    data = pd.read_csv('Samsung Dialog.txt', sep=':', header=None, names=['Speaker', 'Line'])
except FileNotFoundError:
    st.error("🚨 Error: The file 'Samsung Dialog.txt' was not found. Please place it in the root of your repository.")
    st.stop()

cust = data.loc[data['Speaker'] == 'Customer', 'Line'].reset_index(drop=True)
sales = data.loc[data['Speaker'] == 'Sales Agent', 'Line'].reset_index(drop=True)

# Make sure lengths match
min_len = min(len(cust), len(sales))
cust, sales = cust[:min_len], sales[:min_len]

new_data = pd.DataFrame({'Question': cust, 'Answer': sales})

# --- Define Text Preprocessing Function ---
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return "" # Handle NaN or non-string inputs gracefully

    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        # Use str.isalnum() to filter out most punctuation
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    return ' '.join(preprocessed_sentences)

# Preprocess the entire question corpus
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
        try:
            audio = r.listen(source, timeout=5)
            st.info("📝 Transcribing your voice...")
            text = r.recognize_google(audio, language=language_code)
            st.success(f"✅ Transcription complete: {text}")
            return text
        except sr.WaitTimeoutError:
             st.error("❌ No speech detected after 5 seconds. Please try again.")
             return None
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand your voice. Please try again.")
            return None
        except sr.RequestError:
            st.error("⚠️ Could not reach the speech recognition service. Check your internet connection.")
            return None
        except Exception as e:
            st.error(f"⚠️ An audio error occurred. Check PyAudio installation: {e}")
            return None

# --- Streamlit UI ---
st.markdown("<h1 style='text-align:center; color:#0C2D57;'>🎤 ORGANIZATIONAL SPEECH CHATBOT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#F11A7B;'>Built by Adeseye</h4>", unsafe_allow_html=True)

st.markdown("---")
st.write("This chatbot can respond to your text or speech questions based on a company FAQ file. "
         "You can either type your question or click the button to speak.")

# Use a session state variable to ensure the radio button maintains state
if 'input_type' not in st.session_state:
    st.session_state.input_type = "Text"

input_type = st.radio("Choose your input method:", ["Text", "Speech"], key='input_type')

user_input = None

if input_type == "Text":
    user_input = st.chat_input("Ask your question here...")
elif input_type == "Speech":
    if st.button("🎙️ Click to Record Speech"):
        user_input = transcribe_speech()

if user_input:
    # Normalize input for matching and processing
    user_input = user_input.lower()
    
    # Display human input in chat
    with st.chat_message("human"):
        st.write(user_input)

    response = ""
    if user_input in human_greeting:
        response = random.choice(bot_greeting)
    elif user_input in human_exit:
        response = random.choice(bot_farewell)
    else:
        # 1. Preprocess the user's question
        processed_input = preprocess_text(user_input)
        
        # Check if the input yielded any meaningful words after preprocessing
        if not processed_input.strip():
             response = "I'm sorry, I couldn't understand your query. Please try asking a more specific question."
        else:
            # 2. Vectorize the input
            vect_input = tfidf_vectorizer.transform([processed_input])
            
            # 3. Calculate similarity with the corpus
            similarity_scores = cosine_similarity(vect_input, corpus)
            
            # 4. Find the best match
            most_similar_index = np.argmax(similarity_scores)
            
            # Optional: Add a threshold to reject low-confidence answers
            max_score = similarity_scores[0, most_similar_index]
            
            # Use a slightly lower threshold for robustness, e.g., 0.15
            if max_score < 0.15: 
                 response = "I couldn't find a close answer in my knowledge base. Could you rephrase your question?"
            else:
                # 5. Get the corresponding answer
                response = new_data['Answer'].iloc[most_similar_index]

    # Display the bot's response
    with st.chat_message("ai"):
        st.write(response)