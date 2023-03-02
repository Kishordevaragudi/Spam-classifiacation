import pickle
import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st


model = pickle.load(open("Naive_model.pkl","rb"))

st.title("Email Spam Classifier")

message_text = st.text_input("Enter a message for spam evaluation")

def classify_message(model,message):
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])
    return {'label':label,'spam_probability':spam_prob[0][1]}

if message_text !='':
    result =  classify_message(model,message_text)
    st.write(result)