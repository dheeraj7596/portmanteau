import pandas as pd
from sklearn.model_selection import train_test_split


def write_train(df):
    f = open("./port.train", "w")
    for index, row in df.iterrows():
        src = row["source"]
        tgt = row["target"]
        for i, c in enumerate(src):
            f.write(c + " " + tgt[i] + "\n")
        f.write("\n")
    f.close()


def write_val(df):
    f1 = open("./port.dev", "w")
    f2 = open("./port_key.txt", "w")
    for index, row in df.iterrows():
        src = row["source"]
        tgt = row["target"]
        for i, c in enumerate(src):
            f1.write(c + "\n")
            f2.write(c + " " + tgt[i] + "\n")
        f1.write("\n")
        f2.write("\n")
    f1.close()
    f2.close()


if __name__ == "__main__":
    data_path = "/Users/dheerajmekala/Academics/q1/cse256/portmanteau/data/"
    df_1 = pd.read_csv(data_path + "components-blends-blind.csv", delimiter="\t")
    df_2 = pd.read_csv(data_path + "components-blends-knight.csv", delimiter="\t")
    df_all = pd.concat([df_1, df_2])

    train, test = train_test_split(df_all, test_size=0.1, random_state=42)
    train.drop(train.columns[train.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    test.drop(test.columns[test.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    write_train(train)
    write_val(test)
    pass
