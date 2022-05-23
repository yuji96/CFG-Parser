import pickle
from pathlib import Path
from random import sample
from time import time

from nltk.tree import Tree
from tqdm import tqdm

from cky import CKY, build_tree
from reader import read_cleaned_corpus

pwd = Path(__file__).parent
lexical_dict = pickle.loads(pwd.joinpath("stats/lexical_dict.pkl").read_bytes())
syntax_dict = pickle.loads(pwd.joinpath("stats/syntax_dict.pkl").read_bytes())
unary_dict = pickle.loads(pwd.joinpath("stats/unary_dict.pkl").read_bytes())


def predict(gold: Tree) -> Tree:
    a = time()
    gold.chomsky_normal_form()
    # lexical_dict, syntax_dict, unary_dict = rule_as_dict(gold.productions())
    _, backpointer = CKY(gold.leaves(), lexical_dict, syntax_dict, unary_dict,
                         beam=25)
    b = time()
    pred = build_tree(backpointer)
    pred.un_chomsky_normal_form()
    c = time()
    tqdm.write(repr((b - a, c - b)))
    return pred


if __name__ == "__main__":
    golds = sample(read_cleaned_corpus("test"), 10)
    # golds = read_cleaned_corpus("debug")
    Path("tmp/gold.txt").write_text("\n".join(
        [tree.pformat(margin=float("inf")) for tree in golds]))
    preds = []
    for gold in tqdm(golds.copy()):
        pred = predict(gold)
        preds.append(pred)

    Path("tmp/pred.txt").write_text("\n".join(
        [tree.pformat(margin=float("inf")) for tree in preds]))

    # 可視化用
    # chart_count = [[len(tags) for tags in row] for row in chart]
    # plt.imshow(chart_count)
    # plt.show()
    # pprint(chart[0][-1])
    # visible_print(chart)
    # pred_tree.pretty_print()
