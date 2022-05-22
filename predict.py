import pickle
from pathlib import Path

from nltk.tree import Tree
from tqdm import tqdm

from cky import CKY, build_tree
from reader import read_cleaned_corpus

lexical_dict = pickle.loads(Path("stats/lexical_dict.pkl").read_bytes())
syntax_dict = pickle.loads(Path("stats/syntax_dict.pkl").read_bytes())
unary_dict = pickle.loads(Path("stats/unary_dict.pkl").read_bytes())

golds = read_cleaned_corpus("debug")
preds = []
for gold in tqdm(golds.copy()):
    gold: Tree

    gold.chomsky_normal_form()
    chart, backpointer = CKY(gold.leaves(), lexical_dict, syntax_dict, unary_dict,
                             beam=30)
    pred = build_tree(backpointer)

    gold.un_chomsky_normal_form()
    pred.un_chomsky_normal_form()
    preds.append(pred)

Path("tmp/gold.txt").write_text("\n".join(
    [tree.pformat(margin=float("inf")) for tree in golds]))
Path("tmp/pred.txt").write_text("\n".join(
    [tree.pformat(margin=float("inf")) for tree in preds]))

# 可視化用
# chart_count = [[len(tags) for tags in row] for row in chart]
# plt.imshow(chart_count)
# plt.show()
# pprint(chart[0][-1])
# visible_print(chart)
# pred_tree.pretty_print()
