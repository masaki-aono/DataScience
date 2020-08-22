import collections
import numpy as np
import re # regular expression
import os
from nltk import word_tokenize
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from nltk.corpus import stopwords
from wikipedia2vec import Wikipedia2Vec
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import CuDNNLSTM, Input
from keras.layers import Bidirectional
from keras import regularizers
import keras.callbacks
import matplotlib.pyplot as plt

MODEL_FILE = os.path.dirname(__file__) + '/../data/enwiki_20180420_100d.pkl'
wikipedia2vec = Wikipedia2Vec.load(MODEL_FILE)


# Tokenization with NLTK, TF-IDF vectorizer with scikit-learn
def tokenize(text): # returning tokens
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))

    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
    return filtered_tokens

# Loading the corpus
ma_reuters = LazyCorpusLoader(
    'ma_reuters', CategorizedPlaintextCorpusReader, '(training|test).*',
    cat_file='cats.txt', encoding='ISO-8859-2')

# Load MA_Reuters
documents = ma_reuters.fileids()
# print (str(len(documents)) + " total articles")
# extracting training and testing data (document ID)
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
# Training and testing data
train_docs = [ma_reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [ma_reuters.raw(doc_id) for doc_id in test_docs_id]

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([ma_reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([ma_reuters.categories(doc_id) for doc_id in test_docs_id])

maxlen = 20 # 1文書に含まれる層単語数の上限を保持
min_length = 3 # 1単語の文字数の最小値(3文字以上の単語のみ残す)
word_counter = collections.Counter()
docs = [train_docs, test_docs]

for document in docs: # 単語の小文字化と抽出
    num_data = len(document)
    for i in range(num_data):
        text = document[i]
        words = map(lambda word: word.lower(), word_tokenize(text))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
        if len(filtered_tokens) > maxlen:
            maxlen = len(filtered_tokens)
        for word in filtered_tokens:
            word_counter[word] += 1

print("maxlen = ",maxlen)
print(" Word count = ", len(word_counter),' ',type(word_counter))

print("語彙生成 creating vocabulary...")
VOCAB_SIZE = 25000 # Reuters News 最大語彙の設定（これ以上は無視する）
word2index = collections.defaultdict(int)
for wid, word in enumerate(word_counter.most_common(VOCAB_SIZE)):# 頻度順
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v:k for k, v in word2index.items()}
index2word[0] = "_UNK_" # 未知語
print("len(word2index) = ", len(word2index))
print("index2word[1] = ",index2word[1])

print("訓練用データの単語列生成 creating word sequences...")

min_length = 3
cachedStopWords = stopwords.words("english")

xs_train = []
document = train_docs
num_data = len(document)
for i in range(num_data):
    text = document[i]
    words = [x.lower() for x in word_tokenize(text)] # NLTK's word tokenizer
    words = [word for word in words if word not in cachedStopWords]
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
    wids = [word2index[word] for word in filtered_tokens]
    xs_train.append(wids)

X_train = pad_sequences(xs_train, maxlen=maxlen)# パディング (1861単語が最大)
Y_train = train_labels # np_utils.to_categorical(ys)  多値分類なのでワンホットではない!!
print("訓練データ（データ＋ラベル）")
print("X_train",X_train.dtype," ",type(X_train)," ",X_train.shape)
print("Y_train",Y_train.dtype," ",type(Y_train)," ",Y_train.shape)

print("テスト用データの単語列生成 creating word sequences...")
xs_test = []
document = test_docs
num_data = len(document)
for i in range(num_data):
    text = document[i]
    words = [x.lower() for x in word_tokenize(text)] # NLTK's word tokenizer
    words = [word for word in words if word not in cachedStopWords]
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
    wids = [word2index[word] for word in filtered_tokens]
    # wids = [word2index[word] for word in words]
    xs_test.append(wids)

X_test = pad_sequences(xs_test, maxlen=maxlen)# パディング
Y_test = test_labels # np_utils.to_categorical(ys) 多値分類なのでワンホットではない!!
print("テストデータ（データ＋ラベル）")
print("X_test",X_test.dtype," ",type(X_test)," ",X_test.shape)
print("Y_test",Y_test.dtype," ",type(Y_test)," ",Y_test.shape)

Xtrain = X_train
Xtest = X_test
Ytrain = Y_train
Ytest = Y_test
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# 最大語彙サイズ
VOCAB_SIZE = 27000

EMBED_SIZE = 100

embedding_weights = np.zeros((vocab_sz, EMBED_SIZE))
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = wikipedia2vec.get_entity_vector(wikipedia2vec.get_entity(word.capitalize()).title)
    except AttributeError:
        pass

print("Embedding_weight matrix size = ", embedding_weights.shape)

NUM_CLASSES = 55
HIDDEN_LAYER_SIZE = 256

inputs = Input(shape=(maxlen,))
x = Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen,
                    weights=[embedding_weights],
                    trainable=True)(inputs)
x = SpatialDropout1D(0.3)(x)
x = Bidirectional(CuDNNLSTM(HIDDEN_LAYER_SIZE,
                           kernel_regularizer=regularizers.l2(1e-7)))(x)
x = Dense(512,activation="elu",
         kernel_regularizer=regularizers.l2(1e-7))(x)
x = Dropout(0.4)(x)
outputs = Dense(NUM_CLASSES, activation="sigmoid")(x)
model2 = Model(inputs=[inputs], outputs=[outputs])

model2.summary()

model2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["categorical_accuracy"])

NUM_EPOCHS = 20
BATCH_SIZE = 128

fpath = 'h5/Reuters-LSTM-w-{epoch:02d}-{loss:.2f}-{val_loss:.4f}.h5'
callbacks = [
    keras.callbacks.ModelCheckpoint(fpath, monitor='val_loss', save_best_only=True),
]

history2 = model2.fit(
    Xtrain, Ytrain,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    validation_data=(Xtest, Ytest))

# Save model1 and history1
model2.save_weights('h5/MA_Reuters-LSTM-2020-8-17-weights.h5')
model2.save('h5/MA_Reuters-LSTM-2020-8-17.h5')
