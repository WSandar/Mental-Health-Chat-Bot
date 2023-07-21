#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from tensorflow.keras.layers import LSTM


# In[3]:


import pickle


# In[4]:


with open("../Machine Learning/intents.json") as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data['intents'])
df


# In[7]:


df.isnull().sum()


# In[5]:


training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
    
num_classes = len(labels)


# In[7]:


lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)


# In[6]:


vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index 
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences,truncating='post', maxlen = max_len)


# In[8]:


from tensorflow.keras.layers import LSTM, Dropout



model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(64))  # You can adjust the number of units in the LSTM layer as needed
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(num_classes, activation='softmax'))



# In[9]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


epochs = 510
result = model.fit(padded_sequences, np.array(training_labels), epochs = epochs)


# In[11]:


# to save the train model
model.save("chat_model")

import pickle

#to save the fitted tokenizer
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)

#to save the fitted label encoder
with open('label_encoder.pickle','wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol= pickle.HIGHEST_PROTOCOL)


# In[12]:


import colorama
colorama.init()
from colorama import Fore,Style,Back


# In[ ]:


def chat():
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
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User:" + Style.RESET_ALL, end ="")
        inp = input()
        if inp.lower() == "quit":
            break
            
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post', maxlen = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, np.random.choice(i['responses']) )
                
                # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses)) 
                
print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




