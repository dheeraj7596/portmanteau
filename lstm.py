import Levenshtein
import numpy as np
from sentence_getter import SentenceGetter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


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


if __name__ == "__main__":
    data = pickle.load(open("./data/df_lstm.pkl", "rb"))
    embeddings_path = "./data/pretrained_char_emb.txt"

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

    model = get_model(max_len, n_words, n_tags, embedding_mat)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

    preds = []
    true = []
    for i, test in enumerate(X_te):
        p = model.predict(np.array([X_te[i]]))
        t = y_te[i]
        p = np.argmax(p, axis=-1)
        t = np.argmax(t, axis=-1)

        preds.append(get_word(X_te[i], p[0], words, tags))
        true.append(get_word(X_te[i], t, words, tags))

    distance = 0
    for i, word in enumerate(true):
        distance += Levenshtein.distance(word, preds[i])
    print(distance / len(preds))