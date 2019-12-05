import Levenshtein
from portmanteau.sentence_getter import SentenceGetter
from keras.preprocessing.sequence import pad_sequences
import pickle
import kenlm
from keras.models import load_model
from portmanteau.utils import *
from collections import Counter
import numpy as np

lmodel = kenlm.Model('/Users/dheerajmekala/Academics/q1/cse256/portmanteau/data/wordlist_english_filtered_threshold100-kenlm.arpa')

word2idx = {}
tag2idx = {}
words = []
tags = []
len_model = None


def pred_one(X, prediction, words, tag2idx, len_model, lmodel):
    predictions = getTopk(prediction[0], 10, tag2idx)
    candidates = [get_word_by_tag(X, d[1], words) for d in predictions]
    m_scores = [lmodel.score(" ".join(c)) / (float(len(" ".join(c)))) for c in candidates]
    input_len = [len_model.predict([[p[1].index('O')]])[0] if 'O' in p[1] else 30 for p in predictions]
    lstm_len = [(p[1].index('O') - p[1].count('D')) if 'O' in p[1] else 30 for p in predictions]
    len_score = [1 / (1 + (abs(i - l))) for l, i in zip(lstm_len, input_len)]
    for j in range(len(m_scores)):
        # m_scores[j] = predictions[j][0]
        m_scores[j] = m_scores[j] / 8 + predictions[j][0] + len_score[j] / 45
    max_idx = -1
    max_val = -99999
    for ele in enumerate(m_scores):
        if ele[1] > max_val:
            max_val = ele[1]
            max_idx = ele[0]
    return candidates[max_idx]


def predict(data):
    global word2idx, tag2idx, words, tags, len_model, lmodel
    max_len = 30

    if not len_model:
        len_model = pickle.load(open("/Users/dheerajmekala/Academics/q1/cse256/portmanteau/dumps/len_model.pkl", "rb"))

    if not tag2idx:
        tag2idx = pickle.load(open("/Users/dheerajmekala/Academics/q1/cse256/portmanteau/dumps/tag2idx.pkl", "rb"))

    if not word2idx:
        word2idx = pickle.load(open("/Users/dheerajmekala/Academics/q1/cse256/portmanteau/dumps/word2idx.pkl", "rb"))

    if not words:
        words = pickle.load(open("/Users/dheerajmekala/Academics/q1/cse256/portmanteau/dumps/words.pkl", "rb"))

    if not tags:
        tags = pickle.load(open("/Users/dheerajmekala/Academics/q1/cse256/portmanteau/dumps/tags.pkl", "rb"))

    model = load_model("/Users/dheerajmekala/Academics/q1/cse256/portmanteau/dumps/lstm.h5")

    n_words = len(words)
    x = data[0].lower() + "}" + data[1].lower()
    X = [[word2idx[w] for w in x]]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
    p = model.predict(np.array([X[0]]))
    ans = pred_one(X[0], p, words, tag2idx, len_model, lmodel)
    return ans
