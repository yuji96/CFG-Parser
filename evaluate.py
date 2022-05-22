import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from nltk.tree import Tree

from reader import read_cleaned_corpus


def evalb_to_df(text: str):
    indivisual, all_sammary, cutoff_sammary = re.split(r"-- .+ --", text)

    all_sammary = pd.Series(dict(re.findall(r"(.+?)\s+=\s+(.+)", all_sammary)),
                            name="all")
    cutoff_sammary = pd.Series(
        dict(re.findall(r"(.+?)\s+=\s+(.+)", cutoff_sammary)), name="cutoff")
    sammary = pd.concat([all_sammary, cutoff_sammary], axis="columns")

    _, indivisual, _ = re.split(r"={4,}", indivisual, maxsplit=3)
    df = pd.DataFrame([line.split() for line in indivisual.strip().splitlines()],
                      columns=[
                          "id", "len", "stat", "recall", "precision",
                          "matched_bracket", "bracket_gold", "bracket_test",
                          "cross_bracket", "words", "correct_tags", "tag_accuracy"
                      ])
    return df, sammary


def evaluate(gold_trees: list[Tree], pred_trees: list[Tree]):
    evalb = Path(__file__).resolve().parent.joinpath("EVALB/evalb")
    assert evalb.exists(), "evalb コマンドが見つかりません。"

    golds = "\n".join([tree.pformat(margin=float("inf")) for tree in gold_trees])
    preds = "\n".join([tree.pformat(margin=float("inf")) for tree in pred_trees])

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        d.joinpath("gold.txt").write_text(golds)
        d.joinpath("pred.txt").write_text(preds)
        pipe = subprocess.run([str(evalb), d / "gold.txt", d / "pred.txt"],
                              encoding='utf-8', stdout=subprocess.PIPE)
    df, summary = evalb_to_df(pipe.stdout)
    # assert len(df) == len(paths)
    # df["path"] = [path.name for path in paths]
    return df, summary


if __name__ == "__main__":
    trees = read_cleaned_corpus("test")
    df, sammary = evaluate(trees, trees)
    print(df)
