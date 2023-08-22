import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import speech_recognition as sr
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Read and preprocess data
df = pd.read_csv("weather.csv")
df.dropna(inplace=True)

# Encoding
encoder = OrdinalEncoder()
df[['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']] = encoder.fit_transform(
    df[['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']])
enc_s = LabelEncoder()
df['RainTomorrow'] = enc_s.fit_transform(df['RainTomorrow'])

# Feature Selection
selected_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
                     'RainToday']
x = df[selected_features]
y = df['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
bst = XGBClassifier(max_depth=2)
bst.fit(x_train, y_train)

# Initialize the speech recognition recognizer
r = sr.Recognizer()

def predict_weather(user_features, selected_features):
    features = pd.DataFrame([user_features], columns=selected_features)
    prediction = bst.predict(features)[0]
    if prediction == 0:
        return "No rain expected for tomorrow."
    else:
        return "It will likely rain tomorrow. Don't forget to take your umbrella!"

def transcribe_speech():
    with sr.Microphone() as source:
        st.info('Speak now...')
        print("This is only available for English speakers! Thanks for understanding!")
        audio_data = r.record(source, duration=10)
        print("Recognizing...")
        try:
            text = r.recognize_google(audio_data)
            st.write("Transcription:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry I didn't get that or you ran out of time")
            st.write("Sorry I didn't get that/You ran out of time")

def greet():
    st.write("Chatbot: Hi there! Welcome to the Weather Prediction App!")

def main():
    st.set_page_config(page_title="Weather Prediction Application")
    st.title("Chatbot with Text/Speech Input")

    input_type = st.radio("Choose input type:", ["Text", "Speech"])

    greet()

    if input_type == "Text":
        st.write("Chatbot: Please provide the following features:")
        user_input = [st.number_input(f"Enter {feature}:", key=feature) for feature in selected_features]
        submit_button = st.button("Predict Weather")

        if submit_button:
            weather_result = predict_weather(user_input, selected_features)
            st.write("Chatbot:", weather_result)

    elif input_type == "Speech":
        st.info('Please press the "Start Speech Recognition" button and speak...')
        start_button = st.button("Start Speech Recognition")

        if start_button:
            user_input = transcribe_speech()
            if user_input:
                user_input_values = user_input.split(',')
                if len(user_input_values) == len(selected_features):
                    try:
                        user_input_values = list(map(float, user_input_values))
                        weather_result = predict_weather(user_input_values, selected_features)
                        st.write("Chatbot:", weather_result)
                    except ValueError:
                        st.write("Chatbot: Please provide valid numerical values for all selected features.")
                else:
                    st.write("Chatbot: Please provide the correct number of values for all selected features.")

if __name__ == "__main__":
    main()
