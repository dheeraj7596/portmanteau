import Levenshtein


def get_words(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    words = []
    word = ""
    for line in lines:
        ents = line.strip().split()
        if len(ents) == 0:
            if len(word) > 0:
                words.append(word)
            word = ""
            continue
        if ents[1] == "C":
            word += ents[0]
    return words


def compute():
    key_path = "./port_key.txt"
    pred_path = "./port_dev.p1.out"
    words_true = get_words(key_path)
    words_pred = get_words(pred_path)
    distance = 0
    for i, word in enumerate(words_true):
        distance += Levenshtein.distance(word, words_pred[i])
    return distance / len(words_pred)


if __name__ == "__main__":
    ans = compute()
    print(ans)
