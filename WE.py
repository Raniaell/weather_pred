import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sqlite3
import io
import random
import string
import speech_recognition as sr 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pyaudio
#Downoald NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

df= pd.read_csv("weather.csv")
#Drop missing values
df.dropna(inplace = True)
#Encoding
encoder=OrdinalEncoder()
df[['WindGustDir','WindDir9am','WindDir3pm','RainToday']]=encoder.fit_transform(df[['WindGustDir','WindDir9am','WindDir3pm','RainToday']])
enc_s=LabelEncoder()
df['RainTomorrow']= enc_s.fit_transform(df['RainTomorrow'])

#Features Selection
selected_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'RainToday']
x = df[selected_features]
y = df[['RainTomorrow']]
#Train_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
bst = XGBClassifier(max_depth=2)
bst.fit(x_train, y_train)
y_pred = bst.predict(x_test)
#print(classification_report(y_test,y_pred))

#Store database in Sqlite
conn = sqlite3.connect('Weather_Prediction_DB.sqlite')
conn.close()



# Initialize the speech recognition recognizer
r = sr.Recognizer()

def predict_weather(user_features, selected_features):
    features = pd.DataFrame([user_features],
               columns =selected_features)
    prediction = bst.predict(features)[0]
    if prediction == 0:
        return "No rain expected for tomorrow."
    else:
        return "It will likely rain tomorrow. Don't forget to take your umbrella!"

def transcribe_speech():
    with sr.Microphone() as source:
        st.info('Speak now...')
        print("This is only available for english speakers! Thanks for understanding!")
        audio_data = r.record(source, duration=5)
        print("Recognizing...")
        print("this is your recording: ",audio_data)
        try:
            text = r.recognize_google(audio_data)
            print(audio_data.get_wav_data(), text)
            st.write("Transcription:", text)
            return text
        except:
            print("Sorry I didn't get that or you run out of time")
            st.write("Sorry I didn't get that/You run out of time")

# Fonction pour la salutation
def greet():
    st.write("Chatbot: Hi there!, Welcome to the weather Prediction App!")

# Fonction pour collecter les informations de l'utilisateur (textuel)
def collect_user_input_text(key):
    st.write("Chatbot: Sure, please provide me with the following features:")
    st.write(", ".join(selected_features))
    user_features = st.text_input("Provide the values to predict tomorrow's weather:", key=key)
    return user_features

def collect_user_input_speech(key):
    st.info('Speak now...')
    st.write("Chatbot: This is only available for English speakers. Thanks for understanding!")
    user_input = transcribe_speech()
    st.write("Transcription:", user_input)
    return user_input


# Fonction pour prédire la météo
def weather_prediction(user_input):
    try:
        user_features = list(map(float, user_input.split(',')))

        if len(user_features) != len(selected_features):
            st.write("Chatbot: Please provide valid numerical values for all the selected features.")
            return

        # Call the weather prediction function
        weather_result = predict_weather(user_features, selected_features)
        st.write("Chatbot:", weather_result)

        st.write("Thank you for trusting us! Have a lovely day!")
    except ValueError:
        st.write("Chatbot: Please provide valid numerical values for all the selected features.")

# ...
def main():
    st.set_page_config(page_title="Weather Prediction Application")
    st.title("Chatbot with Text/Speech Input")

    input_type = st.radio("Choose input type:", ["Text", "Speech"])
    
    # Initial state
    conversation_state = "greeting"
    user_input_key = None

    while True:
        if conversation_state == "greeting":
            greet()
            conversation_state = "waiting_for_user_input"
        elif conversation_state == "waiting_for_user_input":
            if input_type == "Text":
                user_input_key = 'user_input_text'
                user_input = collect_user_input_text(user_input_key)
            elif input_type == "Speech":
                user_input_key = 'user_input_speech'
                user_input = collect_user_input_text(user_input_key)

            if user_input:
                conversation_state = "predicting_weather"
        elif conversation_state == "predicting_weather":
            weather_prediction(user_input)
            conversation_state = "waiting_for_user_input" 


if __name__ == "__main__":
    main()

