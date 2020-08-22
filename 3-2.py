from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *

# Loading the corpus
ma_reuters = LazyCorpusLoader(
    'ma_reuters', CategorizedPlaintextCorpusReader, '(training|test).*',
    cat_file='cats.txt', encoding='ISO-8859-2')

# Load MA_Reuters
documents = ma_reuters.fileids()
print (str(len(documents)) + " total articles")
# extracting training and testing data (document ID)
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
print (str(len(train_docs_id)) + " training data")
print (str(len(test_docs_id)) + " testing data")
# Training and testing data
train_docs = [ma_reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [ma_reuters.raw(doc_id) for doc_id in test_docs_id]

# print the total number of categories
categories = ma_reuters.categories()
num_categories = len(categories)
print (num_categories, " categories")
print (categories)

from nltk import word_tokenize
import re

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
    return filtered_tokens

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)
print("converted to TF-IDF model")
print("training document dimension ：",vectorised_train_documents.shape)
print("testing document dimension：",vectorised_test_documents.shape)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([ma_reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([ma_reuters.categories(doc_id) for doc_id in test_docs_id])

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras import backend as K
K.set_session(session)
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop
BoW_dimension = vectorised_train_documents.shape[1]
NUM_CLASSES = 55
inputs = Input(shape=(BoW_dimension,))
x = Dense(512, activation='elu')(inputs)
x = Dropout(0.3)(x)
x = Dense(512, activation='elu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='sigmoid')(x)
model = Model(inputs=[inputs], outputs=[outputs])

model.summary()

model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
batch_size = 128
epochs = 40

model.fit(
    vectorised_train_documents.toarray(), train_labels,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(vectorised_test_documents.toarray(), test_labels))
