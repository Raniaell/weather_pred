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
    
   
def weather_prediction(user_input):    
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["Hi!", "How can I help you?", "Hey!", "Hi there!, Welcome to the weather Prediction App!", "Hello!", "I am glad to help with your requests!"]

    while True:
        #user_input = input("You: ")
        
        if user_input.lower() == "exit":
            st.write("Chatbot: Goodbye!")
            break

        if user_input.lower() in GREETING_INPUTS:
            print(random.choice(GREETING_RESPONSES))
            
        elif "weather" in user_input.lower():
            print("Chatbot: Sure, please provide me with the following features:")
            print(", ".join(selected_features))
            #user_features = input("You (provide values separated by commas): ")
            user_features = st.text_input("Provide the values to predict tomorrow's weather: ", key='user_features')
            try:
                user_features = list(map(float, user_features.split(',')))

                if len(user_features) != len(selected_features):
                    raise ValueError()

                # Call the weather prediction function
                weather_prediction = predict_weather(user_features, selected_features)
                st.write("Chatbot:", weather_prediction)

                st.write("Thank you for trusting us! Have a lovely day!")
                break
            except ValueError:
                print("Chatbot: Please provide valid numerical values for all the selected features.")
        else:
            print("Chatbot: Sorry, I'm here to help you with weather predictions. Just ask about the weather!")


    
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

def main():
    st.set_page_config(page_title="Weather Prediction Application")
    st.title("Chatbot with Text/Speech Input")

    # Choose input type: Text or Speech
    input_type = st.radio("Choose input type:", ["Text", "Speech"])

    user_input = ""
    if input_type == "Text":
        user_input = st.text_input("User:")
    elif input_type == "Speech":
        user_input = transcribe_speech()

    if user_input:
        weather_prediction(user_input)
        

if __name__ == "__main__":
    main()