import pandas as pd


def tokenize(sentence: str):
    # nltk のものを使う
    return sentence.split()


def sent2cell(sent):
    n = len(sent)
    cell = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i, word in enumerate(sent):
        cell[i][i + 1] |= {word}
    return cell


def represent(chart):
    n = len(chart)
    df = pd.DataFrame([["" for _ in row] for row in chart])
    for i, row in enumerate(chart):
        for j, cell in enumerate(row):
            if i < j:
                df.iat[n - (j - i), j] = ", ".join(cell)
    print(df)


if __name__ == "__main__":
    sent = tokenize("This is a pen.")
    cell = sent2cell(sent)
    cell[0][3] |= {"hello"}
    represent(cell)
