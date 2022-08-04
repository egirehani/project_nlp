# Library imports
from json import encoder
import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk

# Load trained Pipeline
model = load_model('Project.h5')

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(tweet)
# vocab_size = len(tokenizer.word_index) + 1
# encoded_docs = tokenizer.texts_to_sequences(tweet)
# padded_sequence = pad_sequences(encoded_docs, maxlen=200)

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)


# creating a function for data cleaning
# from custom_tokenizer_function import CustomTokenizer


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']
    print(new_review)
    tw = tokenizer.texts_to_sequences([new_review])
    tw = pad_sequences(tw,maxlen=300)
    predictions = int(model.predict(tw).round().item())
    if predictions==0:
        return render_template('index.html', prediction_text='Senang')
    elif predictions==1:
        return render_template('index.html', prediction_text='Marah')
    elif predictions==2:
        return render_template('index.html', prediction_text='Sedih')
    elif predictions==3:
        return render_template('index.html', prediction_text='Takut')
    elif predictions==4:
        return render_template('index.html', prediction_text='Terkejut')


if __name__ == "__main__":
    app.run(port=3000, debug=True)
