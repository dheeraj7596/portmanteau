from .util import *
from .data_utils import get_viterbi_X


def get_K(n):
    K = {}
    K[-1] = ["*"]
    K[0] = ["*"]
    for i in range(1, n + 1):
        K[i] = ["C", "D"]
    return K


def get_last_tags(K, n, pi, trigram_count_dict, bigram_count_dict):
    """
    Get the last two tags for decoding.
    :param K:
    :param n:
    :param pi:
    :param trigram_count_dict:
    :param bigram_count_dict:
    :return:
    """
    max_uv = -1
    max_u = ""
    max_v = ""
    for u in K[n - 1]:
        for v in K[n]:
            q = trigram_count_dict[(u, v, "STOP")] / bigram_count_dict[(u, v)]
            temp = pi[(n, u, v)] * q
            if temp > max_uv:
                max_uv = temp
                max_u = u
                max_v = v
    return max_u, max_v


def viterbi_alg(X, count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict, vocab, rare_flag):
    """
    Viterbi Algorithm
    :param X:
    :param count_dict:
    :param trigram_count_dict:
    :param bigram_count_dict:
    :param unigram_count_dict:
    :param vocab:
    :param rare_flag:
    :return:
    """
    n = len(X)
    K = get_K(n)
    pi = {}
    bp = {}
    pi[(0, "*", "*")] = 1
    y = [0] * (n + 1)

    for k in range(1, n + 1):
        for v in K[k]:
            for u in K[k - 1]:
                maxi = -1
                for w in K[k - 2]:
                    q = trigram_count_dict[(w, u, v)] / bigram_count_dict[(w, u)]
                    temp_x = X[k - 1]
                    try:
                        temp = vocab[temp_x]
                    except:
                        print("In exception")
                        temp_x = get_token(temp_x, rare_flag)
                    em = emission_probs(count_dict, v, temp_x)
                    temp = pi[(k - 1, w, u)] * q * em
                    if temp > maxi:
                        maxi = temp
                        bp[(k, u, v)] = w
                        pi[(k, u, v)] = maxi

    max_u, max_v = get_last_tags(K, n, pi, trigram_count_dict, bigram_count_dict)
    y[n] = max_v
    y[n - 1] = max_u

    for k in range(n - 2, 0, -1):
        y[k] = bp[(k + 2, y[k + 1], y[k + 2])]
    return y[1:]


if __name__ == "__main__":
    import sys

    n = 0
    path = "./port.counts"
    dev_path = "./port.dev"
    train_path = "./port.train"

    X = get_viterbi_X(dev_path)
    vocab = build_vocab(train_path)

    count_dict = create_count_dict(path)
    trigram_count_dict = create_trigram_count_dict(path)
    bigram_count_dict = create_bigram_count_dict(path)
    unigram_count_dict = create_unigram_count_dict(path)

    lines = []
    for x in X:
        y = viterbi_alg(x, count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict, vocab, n)
        for i, w in enumerate(x):
            line = w + " " + y[i] + "\n"
            lines.append(line)
        lines.append("\n")

    f1 = open("./port_dev.p1.out", "w")
    for line in lines:
        f1.write(line)
    f1.close()
