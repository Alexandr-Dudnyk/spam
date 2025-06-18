import streamlit as st
import sklearn
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer

port_stemmer = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Create a function to generate cleaned data from raw text

def clean_text(text):
    text = word_tokenize(text) # Create tokens
    text= " ".join(text) # Join tokens
    text = [char for char in text if char not in string.punctuation] # Remove punctuations
    text = ''.join(text) # Join the leters
    text = [char for char in text if char not in re.findall(r"[0-9]", text)] # Remove Numbers
    text = ''.join(text) # Join the leters
    text = [word.lower() for word in text.split() if word.lower() not in set(stopwords.words('english'))] # Remove common english words (I, you, we,...)
    text = ' '.join(text) # Join the leters
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)   # error word


st.title('📩 Виявлення СПАМу')

input_sms = st.text_input("Введіть текст повідомлення, щоб перевірити, чи є воно спамом.")

if st.button('Прогнозувати'):

    if input_sms == "":
        st.header('✍️ Введіть повідомлення:')

    else:

        # 1. Preprocess
        transform_text = clean_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transform_text])

        # 3. Prediction
        result = model.predict(vector_input)

        # 4. Display

        if result == 1:
            st.header("❌ Це СПАМ")
        else:
            st.header("✅ Це не спам")
