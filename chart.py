import pandas as pd


def sent2cell(sent):
    """文を 2 次元配列化する。

    CKY チャートを最下段を埋めた状態で返す。

    Parameters
    ----------
    sent : list[str]

    Returns
    -------
    cell : list[list[set[str]]]

    """
    n = len(sent)
    cell = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i, word in enumerate(sent):
        cell[i][i + 1] |= {word}
    return cell


def represent(chart):
    """CKY チャートをいい感じに表示する。"""
    n = len(chart)
    df = pd.DataFrame([["" for _ in row] for row in chart])
    for i, row in enumerate(chart):
        for j, cell in enumerate(row):
            if i < j:
                df.iat[n - (j - i), j] = ", ".join(cell)
    print(df)
