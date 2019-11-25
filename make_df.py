import pandas as pd
import numpy as np


def make_df(df):
    dic = {}
    dic["Sentence #"] = []
    dic["Word"] = []
    dic["Tag"] = []

    i = 0
    for index, row in df.iterrows():
        src = row["source"]
        tgt = row["target"]
        src = src.strip()
        tgt = tgt.strip()
        assert len(src) == len(tgt)
        for j, ch in enumerate(src):
            dic["Sentence #"].append(i)
            dic["Word"].append(ch)
            dic["Tag"].append(tgt[j])
        i += 1
    return pd.DataFrame(dic)

if __name__ == "__main__":
    data_path = "/Users/dheerajmekala/Academics/q1/cse256/portmanteau/data/"
    df_1 = pd.read_csv(data_path + "components-blends-blind.csv", delimiter="\t")
    df_2 = pd.read_csv(data_path + "components-blends-knight.csv", delimiter="\t")
    df_all = pd.concat([df_1, df_2])
    df = make_df(df_all)
    pass
