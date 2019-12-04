import Levenshtein
import numpy as np
from sentence_getter import SentenceGetter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras import optimizers
import kenlm
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
import heapq
from collections import defaultdict 
from keras.models import load_model
from utils import *
from collections import Counter

lmodel = kenlm.Model('./data/wordlist_english_filtered_threshold100-kenlm.arpa')

model = None
word2idx = {}
tag2idx = {}
words = []
tags = []
len_model = None

# def get_model_crf(max_len, n_words, n_tags, embedding_mat, crf):
#     input = Input(shape=(max_len,))
#     model = Embedding(input_dim=n_words, weights=[embedding_mat], output_dim=50, input_length=max_len)(input)
#     model = Dropout(0.1)(model)
#     model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
#     model = TimeDistributed(Dense(100, activation="relu"))(model)  # softmax output layer
#     out = crf(model)
#     model = Model(input, out)
#     return model

def train_len_model():
	def get_len(row):
	    return Counter(list(row))['C']
	df = pd.read_csv('./data/components-blends-knight.csv',sep='\t',index_col=0)
	df["slen"]=df.source.apply(len)
	df["tlen"]=df.target.apply(get_len)
	df["ratio"]=df["slen"]/df["tlen"]
	len_model = BayesianRidge(verbose=True, compute_score=True)
	X=df["slen"].values.reshape(-1,1)
	y=df["tlen"].values
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test=train_test_split(X, y)
	len_model.fit(X_train, y_train)
	return len_model


def get_model(max_len, n_words, n_tags, embedding_mat):
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words, weights=[embedding_mat], output_dim=50, input_length=max_len)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
    model = Model(input, out)
    return model


def get_embedding_matrix(embeddings_path, word2idx):
    embedding_vectors = {}
    with open(embeddings_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            char = line_split[0]
            embedding_vectors[char] = vec

    embedding_matrix = np.zeros((len(word2idx), 50))
    for char in word2idx:
        embedding_vector = embedding_vectors.get(char)
        if embedding_vector is not None:
            embedding_matrix[word2idx[char]] = embedding_vector
    return embedding_matrix


def get_word(X, y, words, tags):
    ans = ""
    for i, ch in enumerate(X):
        if tags[y[i]] == "C":
            ans += words[ch]
    return ans


def predict(data):
    global model, word2idx, tag2idx, words, tags, len_model
    max_len = 30

    if not model:
        model = load_model("./lstm.h5")

    if not word2idx:
        word2idx = pickle.load(open("./word2idx.pkl", "rb"))

    if not tag2idx:
        tag2idx = pickle.load(open("./tag2idx.pkl", "rb"))

    if not words:
        words = pickle.load(open("./words.pkl", "rb"))

    if not tags:
        tags = pickle.load(open("./tags.pkl", "rb"))

    if not len_model:
    	len_model = train_len_model()

    n_words = len(words)
    x = data[0].lower() + "}" + data[1].lower()
    X = [[word2idx[w] for w in x]]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
    p = model.predict(np.array([X[0]]))
    # p = np.argmax(p, axis=-1)
    ans = pred_one(X[0], p, words)
    return ans


def pred_one(X, prediction, words):
    global lmodel, len_model

    predictions = getTopk(prediction[0], 10)
    candidates = [get_word_by_tag(X, d[1], words) for d in predictions]
    m_scores = [lmodel.score(" ".join(c)) / (float(len(" ".join(c)))) for c in candidates]
    input_len = [len_model.predict([[p[1].index('O')]])[0] if 'O' in p[1] else 30 for p in predictions]
    lstm_len = [(p[1].index('O') - p[1].count('D')) if 'O' in p[1] else 30 for p in predictions]
    len_score = [1/(1+(abs(i-l))) for l,i in zip(lstm_len,input_len)]
    for j in range(len(m_scores)):
        m_scores[j] = m_scores[j]/8 + predictions[j][0] + len_score[j]/45
    max_idx = -1
    max_val = -99999
    for ele in enumerate(m_scores):
        if ele[1] > max_val:
            max_val = ele[1]
            max_idx = ele[0]
    return candidates[max_idx]


if __name__ == "__main__":
    data = pickle.load(open("./data/df_lstm.pkl", "rb"))
    embeddings_path = "./data/pretrained_char_emb.txt"

    # data = ["gay", "baby"]
    # ans = predict(data)
    words = list(set(data["Word"].values))
    words.append("$")
    n_words = len(words)
    tags = list(set(data["Tag"].values))
    tags.append("O")
    n_tags = len(tags)
    getter = SentenceGetter(data)
    sentences = getter.sentences
    max_len = 30
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    embedding_mat = get_embedding_matrix(embeddings_path, word2idx)

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
    # print(X_tr[:5],y_tr[:5])
    # crf = CRF(n_tags)
    # model = get_model_crf(max_len, n_words, n_tags, embedding_mat, crf)
    # print(model.summary())
    # model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    # history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

    # model = get_model(max_len, n_words, n_tags, embedding_mat)
    # print(model.summary())
    # model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
    # history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=1, validation_split=0.1, verbose=1)
    model = load_model("./keras_jupyper.h5")
    len_model = train_len_model()
    preds = []
    true = []
    for i, test in enumerate(X_te):
        t = y_te[i]
        t = np.argmax(t, axis=-1)
        p = model.predict(np.array([X_te[i]]))
        final_pred = pred_one(X_te[i], p, words)
        preds.append(final_pred)
        final_true = get_word(X_te[i], t, words, tags)
        true.append(final_true)
        if final_pred == final_true:
            print(final_pred)

    distance = 0
    for i, word in enumerate(true):
        distance += Levenshtein.distance(word, preds[i])
    print(distance / len(preds))
    # model.save("./lstm.h5")
