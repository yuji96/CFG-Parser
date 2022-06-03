import re
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd
from nltk.tree import Tree
from tqdm import tqdm

from counter import rule_as_dict, to_chomsky_rules
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


def evaluate(gold_trees: list[Tree], pred_trees: list[Tree], gold_path=None,
             pred_path=None):
    evalb = Path(__file__).resolve().parent.joinpath("../EVALB/evalb")
    assert evalb.exists(), "evalb コマンドが見つかりません。"

    golds = "\n".join([tree.pformat(margin=float("inf")) for tree in gold_trees])
    preds = "\n".join([tree.pformat(margin=float("inf")) for tree in pred_trees])

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        d.joinpath("gold.txt").write_text(golds)
        d.joinpath("pred.txt").write_text(preds)
        subprocess.run([str(evalb), d / "gold.txt", d / "pred.txt"],
                       encoding='utf-8')

    pwd = Path(__file__).parent
    if isinstance(gold_path, str):
        pwd.joinpath(gold_path).write_text(golds)
    if isinstance(pred_path, str):
        pwd.joinpath(pred_path).write_text(preds)


if __name__ == "__main__":
    import pickle
    from random import sample, seed

    from cky import CKY

    seed(0)

    pwd = Path(__file__).parent
    lexical_dict = pickle.loads(
        pwd.joinpath("../stats/lexical_markov2.pkl").read_bytes())
    syntax_dict = pickle.loads(
        pwd.joinpath("../stats/syntax_markov2.pkl").read_bytes())
    unary_dict = pickle.loads(
        pwd.joinpath("../stats/unary_markov2.pkl").read_bytes())

    # golds = sample(read_cleaned_corpus("test"), 10)
    golds = [Tree.fromstring(Path("../data/failure/2.clean").read_text())]

    preds = [
        CKY(gold.leaves(), lexical_dict, syntax_dict, unary_dict, beam=20)
        for gold in tqdm(deepcopy(golds))
    ]
    # evaluate(golds, preds)
