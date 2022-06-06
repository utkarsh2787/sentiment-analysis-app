from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from tensorflow.keras.models import Sequential

# Keras

from keras.models import load_model
import tensorflow as tf
MODEL_PATH = 'C:/Users/User/Downloads/model.h5'
model=load_model(MODEL_PATH)

# Load your trained model



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
df=pd.read_csv('C:/Users/User/Downloads/Twitter_Data.csv')
df.dropna(inplace=True)
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
df['clean_text'] = df['clean_text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer(language='english')
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join(stemmer.stem(x) for x in x.split() if x not in stop_words))
import string
arr=[]
for a in string.punctuation:
    arr.append(a)
df['clean_text']=df['clean_text'].apply(lambda x: ' '.join(w for w in x.split() if w not in arr))
df['clean_text']=df['clean_text'].apply(lambda x: ' '.join(w for w in x.split() if w[0]!='@'))
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=1000, split=' ')
tokenizer.fit_on_texts(df['clean_text'].values)



# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras

#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(text, model):

    text = ' '.join(x.lower() for x in text.split())
    text = ' '.join(x for x in text.split() if x not in stop_words)
    text = ' '.join(stemmer.stem(x) for x in text.split() if x not in stop_words)
    text = ' '.join(w for w in text.split() if w not in arr)
    text = ' '.join(w for w in text.split() if w[0] != '@')
    lmp = {'data': [text]}
    df_2 = pd.DataFrame(lmp)
    op = tokenizer.texts_to_sequences(df_2['data'])

    up=len(op[0])
    arrlo=[]
    for a in range(39-up):
        arrlo.append(0)
    ff=0
    for a in range(39-up,39):
        arrlo.append(op[0][ff])
        ff=ff+1
    arrop=[arrlo]




    preds = model.predict(arrop)
    return preds

@app.route('/', methods=['GET','POST'])
def main():
    # Main page
    if request.method=="POST":
        inp=request.form.get('inp')
        print(inp)
        opoo=model_predict(inp,model).argmax(axis=-1)
        if opoo==0:
            return render_template('home.html',message='Negative 😢😢😢')
        elif opoo==1:
            return render_template('home.html', message='Neutral 😐😐😐')
        else:
            return render_template('home.html', message='Positive 😊😊😊')

    return render_template('home.html')




if __name__ == '__main__':
    app.run(debug=True)