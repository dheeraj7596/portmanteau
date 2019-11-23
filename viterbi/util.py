from .data_utils import get_token


def create_count_dict(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    count_dict = {}
    for line in lines:
        entities = line.strip().split()
        if entities[1] == "WORDTAG":
            # (x, y) is the key
            count_dict[(entities[3], entities[2])] = int(entities[0])
        elif entities[1] == "1-GRAM":
            count_dict[entities[2]] = int(entities[0])
    return count_dict


def emission_probs(count_dict, y, x):
    try:
        denominator = count_dict[y]
        numerator = count_dict[(x, y)]
        return numerator / denominator
    except:
        return 0


def create_unigram_count_dict(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    unigram_count_dict = {}

    for line in lines:
        entities = line.strip().split()
        if entities[1] == "1-GRAM":
            unigram_count_dict[(entities[2])] = int(entities[0])
    return unigram_count_dict


def create_bigram_count_dict(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    bigram_count_dict = {}

    for line in lines:
        entities = line.strip().split()
        if entities[1] == "2-GRAM":
            bigram_count_dict[(entities[2], entities[3])] = int(entities[0])
    return bigram_count_dict


def create_fourgram_count_dict(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    fourgram_count_dict = {}

    for line in lines:
        entities = line.strip().split()
        if entities[1] == "4-GRAM":
            fourgram_count_dict[(entities[2], entities[3], entities[4], entities[5])] = int(entities[0])
    return fourgram_count_dict


def create_trigram_count_dict(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    trigram_count_dict = {}

    for line in lines:
        entities = line.strip().split()
        if entities[1] == "3-GRAM":
            trigram_count_dict[(entities[2], entities[3], entities[4])] = int(entities[0])
    return trigram_count_dict


def build_vocab(path):
    vocab = {}
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        if line == "\n":
            continue
        entities = line.strip().split()
        vocab[entities[0]] = 1
    return vocab


def simple_tagger(x, count_dict, vocab, flag):
    y_possibles = ["I-GENE", "O"]
    try:
        temp = vocab[x]
    except:
        x = get_token(x, flag)

    p1 = emission_probs(count_dict, y_possibles[0], x)
    p2 = emission_probs(count_dict, y_possibles[1], x)

    if p1 > p2:
        return y_possibles[0]
    else:
        return y_possibles[1]


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1])
    f = open("./gene.dev", "r")
    lines = f.readlines()
    f.close()
    counts_path = "./gene.counts"
    count_dict = create_count_dict(counts_path)
    vocab = build_vocab("./gene.train_modified")

    for i, line in enumerate(lines):
        if line == "\n":
            continue
        line = line.strip()
        tag = simple_tagger(line, count_dict, vocab, n)
        lines[i] = line + " " + tag + "\n"

    f1 = open("./gene_dev.p1.out", "w")
    for line in lines:
        f1.write(line)
    f1.close()
