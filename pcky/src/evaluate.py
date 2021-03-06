import re
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd
from nltk.tree import Tree
from tqdm import tqdm

from counter import swap_dict
from reader import read_cleaned_corpus

pwd = Path(__file__).parent


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
    evalb_dir = Path(__file__).joinpath("../../EVALB").resolve()
    evalb = evalb_dir.joinpath("evalb")
    assert evalb.exists(), f"evalb コマンドが見つかりません。\n{evalb}"

    golds = "\n".join([tree.pformat(margin=float("inf")) for tree in gold_trees])
    preds = "\n".join([tree.pformat(margin=float("inf")) for tree in pred_trees])

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        d.joinpath("gold.txt").write_text(golds)
        d.joinpath("pred.txt").write_text(preds)
        subprocess.run([
            str(evalb), d / "gold.txt", d / "pred.txt", "-p", evalb_dir / "new.prm"
        ], encoding='utf-8')

    if isinstance(gold_path, str):
        pwd.joinpath(gold_path).write_text(golds)
    if isinstance(pred_path, str):
        pwd.joinpath(pred_path).write_text(preds)


def load_dict(suffix, prob=True):
    lexical_prob = pickle.loads(
        pwd.joinpath(f"../stats/lexical_{suffix}.pkl").read_bytes())
    binary_prob = pickle.loads(
        pwd.joinpath(f"../stats/binary_{suffix}.pkl").read_bytes())
    unary_prob = pickle.loads(
        pwd.joinpath(f"../stats/unary_{suffix}.pkl").read_bytes())
    if prob:
        return swap_dict(lexical_prob, binary_prob, unary_prob)
    else:
        return lexical_prob, binary_prob, unary_prob


if __name__ == "__main__":
    import pickle
    from random import sample, seed

    from cky import CKY

    seed(0)

    lexical_dict, syntax_dict, unary_dict = load_dict("markov_2-1", prob=True)

    golds = sample(read_cleaned_corpus("test"), 10)
    # golds = [Tree.fromstring(pwd.joinpath("../data/tmp.clean").read_text())]
    preds = [
        CKY(gold.leaves(), lexical_dict, syntax_dict, unary_dict, beam=30)
        for gold in tqdm(deepcopy(golds))
    ]
    # preds = [to_un_chomsky(p) for p in preds]
    [p.un_chomsky_normal_form() for p in preds]
    evaluate(golds, preds, "gold.txt", "pred.txt")
