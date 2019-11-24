from viterbi.util import *
from viterbi.data_utils import get_viterbi_X

count_dict = {}
fourgram_count_dict = {}
trigram_count_dict = {}
bigram_count_dict = {}
unigram_count_dict = {}
unigram_dict = {}
vocab = None
n = 0


def get_K(n):
    K = {}
    K[-2] = ["*"]
    K[-1] = ["*"]
    K[0] = ["*"]
    for i in range(1, n + 1):
        K[i] = ["C", "D"]
    return K


def smooth_prob(fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict, unigram_dict, w, x, u,
                v):
    """
    Smooths the estimates.
    :param fourgram_count_dict:
    :param trigram_count_dict:
    :param bigram_count_dict:
    :param unigram_count_dict:
    :param unigram_dict:
    :param w:
    :param x:
    :param u:
    :param v:
    :return:
    """
    try:
        four = fourgram_count_dict[(w, x, u, v)] / trigram_count_dict[(w, x, u)]
    except:
        four = 0
    try:
        three = trigram_count_dict[(x, u, v)] / bigram_count_dict[(x, u)]
    except:
        three = 0
    try:
        two = bigram_count_dict[(u, v)] / unigram_count_dict[(u)]
    except:
        two = 0
    try:
        one = unigram_dict[(v)]
    except:
        one = 0

    return lambda1 * four + lambda2 * three + lambda3 * two + lambda4 * one


def get_last_tags(K, n, pi, four_gram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict):
    """
    Outputs the last three tags for decoding.
    :param K:
    :param n:
    :param pi:
    :param four_gram_count_dict:
    :param trigram_count_dict:
    :param bigram_count_dict:
    :param unigram_count_dict:
    :return:
    """
    max_xuv = -1
    max_x = ""
    max_u = ""
    max_v = ""
    for x in K[n - 2]:
        for u in K[n - 1]:
            for v in K[n]:
                try:
                    q = smooth_prob(four_gram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict,
                                    unigram_dict, x, u, v, "STOP")
                except:
                    continue
                temp = pi[(n, x, u, v)] * q
                if temp > max_xuv:
                    max_xuv = temp
                    max_x = x
                    max_u = u
                    max_v = v
    return max_x, max_u, max_v


def viterbi_alg(X, count_dict, fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict,
                unigram_dict, vocab, rare_flag):
    """
    4-gram viterbi algorithm
    :param X:
    :param count_dict:
    :param fourgram_count_dict:
    :param trigram_count_dict:
    :param bigram_count_dict:
    :param unigram_count_dict:
    :param unigram_dict:
    :param vocab:
    :param rare_flag:
    :return:
    """
    n = len(X)
    K = get_K(n)
    pi = {}
    bp = {}
    pi[(0, "*", "*", "*")] = 1
    y = [0] * (n + 1)

    for k in range(1, n + 1):
        for v in K[k]:
            for u in K[k - 1]:
                for x in K[k - 2]:
                    maxi = -1
                    for w in K[k - 3]:
                        q = smooth_prob(fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict,
                                        unigram_dict, w, x, u, v)
                        temp_x = X[k - 1]
                        try:
                            temp = vocab[temp_x]
                        except:
                            print("In exception")
                            temp_x = get_token(temp_x, rare_flag)
                        em = emission_probs(count_dict, v, temp_x)
                        temp = pi[(k - 1, w, x, u)] * q * em
                        if temp > maxi:
                            maxi = temp
                            bp[(k, x, u, v)] = w
                            pi[(k, x, u, v)] = maxi

    max_x, max_u, max_v = get_last_tags(K, n, pi, fourgram_count_dict, trigram_count_dict, bigram_count_dict,
                                        unigram_count_dict)
    y[n] = max_v
    y[n - 1] = max_u
    y[n - 2] = max_x

    for k in range(n - 3, 0, -1):
        y[k] = bp[(k + 3, y[k + 1], y[k + 2], y[k + 3])]
    return y[1:]


def set_global_vars():
    global count_dict, fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict, unigram_dict, vocab, n, lambda1, lambda2, lambda3, lambda4

    lambda1 = 1
    lambda2 = 0
    lambda3 = 0
    lambda4 = 0
    path = "./port.counts"
    train_path = "./port.train"
    if not count_dict:
        count_dict = create_count_dict(path)
    if not fourgram_count_dict:
        fourgram_count_dict = create_fourgram_count_dict(path)
    if not trigram_count_dict:
        trigram_count_dict = create_trigram_count_dict(path)
    if not bigram_count_dict:
        bigram_count_dict = create_bigram_count_dict(path)
    if not unigram_count_dict:
        unigram_count_dict = create_unigram_count_dict(path)
    if not unigram_dict:
        count = 0
        for p in unigram_count_dict:
            count += unigram_count_dict[p]

        for p in unigram_count_dict:
            unigram_dict[p] = unigram_count_dict[p] / count
    if not vocab:
        vocab = build_vocab(train_path)
    n = 0


def predict(data):
    global count_dict, fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict, unigram_dict, vocab, n
    set_global_vars()
    x = data[0] + "}" + data[1]
    y = viterbi_alg(x, count_dict, fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict,
                    unigram_dict, vocab, n)
    ans = ""
    for i, p in enumerate(y):
        if p == "C":
            ans += x[i]
    return ans


if __name__ == "__main__":
    import sys, math

    n = 0

    path = "./port.counts"
    dev_path = "./port.dev"
    train_path = "./port.train"

    X = get_viterbi_X(dev_path)
    vocab = build_vocab(train_path)

    count_dict = create_count_dict(path)
    fourgram_count_dict = create_fourgram_count_dict(path)
    trigram_count_dict = create_trigram_count_dict(path)
    bigram_count_dict = create_bigram_count_dict(path)
    unigram_count_dict = create_unigram_count_dict(path)
    unigram_dict = {}
    count = 0
    for p in unigram_count_dict:
        count += unigram_count_dict[p]

    for p in unigram_count_dict:
        unigram_dict[p] = unigram_count_dict[p] / count

    lambda1 = 1
    lambda2 = 0
    lambda3 = 0
    lambda4 = 0

    lines = []
    for x in X:
        y = viterbi_alg(x, count_dict, fourgram_count_dict, trigram_count_dict, bigram_count_dict, unigram_count_dict,
                        unigram_dict, vocab, n)
        for i, w in enumerate(x):
            line = w + " " + y[i] + "\n"
            lines.append(line)
        lines.append("\n")

    f1 = open("./port_dev.p1.out", "w")
    for line in lines:
        f1.write(line)
    f1.close()
