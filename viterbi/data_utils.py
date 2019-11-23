from collections import defaultdict
import re


def get_token(word, flag):
    """
    Grouping rare words into classes
    :param word:
    :param flag:
    :return:
    """
    if flag == 1:
        return "_RARE_"
    elif flag == 2:
        if bool(re.search(r'\d', word)):
            return "AlphaNum"
        else:
            return "oThEr"
    elif flag == 3:
        if word[-3:] == "ing":
            return "enDiNg"
        else:
            return "oThEr"


def get_rare_words(flag, threshold=5):
    """
    Method to get rare words.
    :param flag:
    :param threshold:
    :return:
    """
    word_freq = defaultdict(int)
    f = open("./gene.train", "r")
    lines = f.readlines()
    for line in lines:
        entities = line.strip().split()
        for e in entities:
            word_freq[e] += 1

    word_freq_sorted = sorted(word_freq.items(), key=lambda x: x[1])
    rare_words_dict = {}
    for word, freq in word_freq_sorted:
        if freq < threshold:
            rare_words_dict[word] = get_token(word, flag)
        else:
            break
    f.close()
    return rare_words_dict


def modify_train(rare_flag):
    """
    Modifies training data by grouping rare words into informative word classes
    :param rare_flag:
    :return:
    """
    rare_words_dict = get_rare_words(rare_flag)
    f = open("./gene.train", "r")
    lines = f.readlines()
    f.close()
    for ind, line in enumerate(lines):
        entities = line.strip().split()
        flag = 0
        for i, e in enumerate(entities):
            try:
                temp = rare_words_dict[e]
                entities[i] = temp
                flag = 1
            except:
                continue
        if flag == 1:
            lines[ind] = " ".join(entities) + "\n"

    f1 = open("./gene.train_modified", "w")
    for line in lines:
        f1.write(line)
    f1.close()


def get_viterbi_X(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    X = []
    temp = []
    for line in lines:
        if line == "\n":
            if len(temp) > 0:
                X.append(temp)
                temp = []
            continue
        line = line.strip()
        temp.append(line)
    return X


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    modify_train(n)
