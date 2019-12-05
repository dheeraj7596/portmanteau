import heapq
import numpy as np


def updatestr(s, i, ch):
    list1 = list(s)
    list1[i] = ch
    str1 = ''.join(list1)
    return str1


def get_word_by_tag(X, y, words):
    ans = ""
    for i, ch in enumerate(X):
        if y[i] == "C":
            ans += words[ch]
    return ans


def getTopk(m, k, r_mapping):
    mapping = {}
    for tag in r_mapping:
        mapping[r_mapping[tag]] = tag
    best_seq = ""
    best_prob = 1.0
    best_idx = np.argmax(m, axis=-1)
    for i in range(30):
        best_seq += mapping[best_idx[i]]
        best_prob *= m[i][best_idx[i]]
    heap = [(-1 * best_prob, best_seq)]
    heapq.heapify(heap)

    result = []
    added = set()
    while k > 0:
        top = heapq.heappop(heap)
        result += [(top[0] * -1, top[1])]
        added.add(top[1])
        k -= 1
        prob = -1 * top[0]
        seq = top[1]
        curr_prob = prob
        curr_seq = seq
        for i in range(30):
            for j in range(3):
                curr_seq = updatestr(curr_seq, i, mapping[j])
                if curr_seq in added:
                    continue
                curr_prob = prob * m[i][j] / m[i][r_mapping[seq[i]]]
                heapq.heappush(heap, (-1 * curr_prob, curr_seq))
                curr_seq = seq
    return result
