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




#raw document example（'coffee category')
# Documents in a category
category_docs = ma_reuters.fileids("retail");
document_id = category_docs[0] # The first document
# print the inside document
print (ma_reuters.raw(document_id))




from nltk import word_tokenize

import re # regular expression

def tokenize(text): # returning tokens
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
    return filtered_tokens

def preprocess(docs):
    lst=[]
    for doc in docs:
        lst.append(' '.join(tokenize(doc)))
    return lst
import torch
from sentence_transformers import SentenceTransformer
# fit_transform
model = SentenceTransformer('bert-base-nli-mean-tokens')
vectorised_train_documents = model.encode(preprocess(train_docs))
# transform
vectorised_test_documents = model.encode(preprocess(test_docs))
print("converted to BERT model")
print("training document dimension ：",vectorised_train_documents.shape)
print("testing document dimension：",vectorised_test_documents.shape)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([ma_reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([ma_reuters.categories(doc_id) for doc_id in test_docs_id])

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# multi-class, multi-label classification and prediction
OVR_classifier = OneVsRestClassifier(LinearSVC(random_state=41, max_iter=10000))
OVR_classifier.fit(vectorised_train_documents, train_labels)
OVR_predictions = OVR_classifier.predict(vectorised_test_documents)

import numpy as np
# Jaccard coefficient
from sklearn.metrics import jaccard_score
print ("Jaccard coef:",np.round(jaccard_score(test_labels, OVR_predictions, average='samples'),3))

# Hamming Loss
from sklearn.metrics import hamming_loss
print ("Hamming Loss:",np.round(hamming_loss(test_labels, OVR_predictions),3))
