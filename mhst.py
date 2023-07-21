import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pickle
import time

st.title('Mental Health Chatbot')

with open("../Machine Learning/intents.json") as datafile:
    data = json.load(datafile)
    df = pd.DataFrame(data['intents'])

#load trained model
model = keras.models.load_model('chat_model')

#load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#load label encoder object
with open('label_encoder.pickle','rb') as enc:
    lbl_encoder = pickle.load(enc)

#parameters
max_len = 20  

if "message" not in st.session_state:
    st.session_state.message=[]
    
for msg in st.session_state.message:
    with st.chat_message(msg["role"]):
         st.markdown(msg["content"])


def predict_sentiment(user_in):
                if user_in is None:
                    return None

                user_in = user_input.lower()

                result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating = 'post', maxlen = max_len))
               
                return result            
 
  
    
# def get_key():
#     return int(time.time())

count = 50
counter = 0

while counter < count :        
        #user_input = st.chat_input()
            input_key = f"chat_input_{counter}"
            user_input = st.chat_input("Say something",key = input_key )
            if user_input == "quit":
                    break
        
            if user_input:
                with st.chat_message("user"): 
                        if user_input != None :  
                            st.markdown(user_input)

                            st.session_state.message.append({"role":"user","content":user_input})


            result = predict_sentiment(user_input)

            if user_input is not None :

                    tag = lbl_encoder.inverse_transform([np.argmax(result)])

                    for i in data['intents']:
                        if i['tag'] == tag:
                            with st.chat_message("assistant"):
                                out = np.random.choice(i['responses'])
                                st.markdown(out)

                            st.session_state.message.append({"role":"assistant","content":out})

            counter += 1 
           