import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sqlite3
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr 
import nltk
from nltk.stem import WordNetLemmatizer
import pyaudio
#Downoald NLTK resources
nltk.download('popular', quiet=True)
nltk.download('nps_chat',quiet=True)
nltk.download('punkt') 
nltk.download('wordnet')

df= pd.read_csv("weather.csv")
#Drop missing values
df.dropna(inplace = True)
#Encoding
encoder=OrdinalEncoder()
df[['WindGustDir','WindDir9am','WindDir3pm','RainToday']]=encoder.fit_transform(df[['WindGustDir','WindDir9am','WindDir3pm','RainToday']])
enc_s=LabelEncoder()
df['RainTomorrow']= enc_s.fit_transform(df['RainTomorrow'])
df
#Features Selection
x = df.drop(['RainTomorrow'],axis=1)
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

#Chatbot_SpeechRecognition
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
# To Recognise input type as QUES. 
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    
#Reading in the input_corpus
with open('weather.csv','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#colour palet
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))

# Generating response and processing 
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
    
#Recording voice input using microphone 
def sp_rg():
    file = "file.mp3"
    flag=True
    fst="My name is X. I will answer your queries about Weather. If you want to exit, say Bye"
    tts = gTTS(fst, lang='en') 
    tts.save(file)
    os.system("mpg123 " + file )
    r = sr.Recognizer()
    prYellow(fst)

    while(flag==True):
        with sr.Microphone() as source:
            audio= r.record(source, duration=10)

        try:
            user_response = format(r.recognize_google(audio))
            print("\033[91m {}\033[00m" .format("YOU SAID : "+user_response))
        except sr.UnknownValueError:
            prYellow("Oops! Didn't catch that")
            pass
        
        #user_response = input()
        #user_response=user_response.lower()
        clas=classifier.classify(dialogue_act_features(user_response))
        if(clas!='Bye'):
            if(clas=='Emotion'):
                flag=False
                prYellow("X: You are welcome..")
            else:
                if(greeting(user_response)!=None):
                    print("\033[93m {}\033[00m" .format("X: "+greeting(user_response)))
                else:
                    print("\033[93m {}\033[00m" .format("X: ",end=""))
                    res=(response(user_response))
                    prYellow(res)
                    sent_tokens.remove(user_response)
                    # Retry generating speech with error handling
                    for _ in range(3):  # You can adjust the number of retries
                        try:
                            tts = gTTS(res, 'en')
                            tts.save(file)
                            os.system("mpg123 " + file)
                            break  # Successful, break out of retry loop
                        except Exception as e:
                            print("Error:", e)
                            print("Retrying...")
        else:
            flag=False
            prYellow("X: Bye! take care..")

def app():
    st.title("Weather Prediction Chatbot")
    st.write("Welcome the weather prediction application")

    user_input = st.text_input("You: ", "")
    if st.button("Ask"):
        st.write("X: ", end="")
        response_text=response(user_input)
        st.write(response_text)

if __name__ == "__main__":
    app()
    sp_rg()