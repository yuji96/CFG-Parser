from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
from nltk.tree import Tree

from cky import CKY
from counter import rule_as_dict

tree: Tree = Tree.fromstring(Path("example/0200-1.txt").read_text())
margin = float("inf")
# Path("tmp/gold.txt").write_text(tree.pformat(margin))

tree.chomsky_normal_form()
lexical_dict, syntax_dict, unary_dict = rule_as_dict(tree.productions())

chart, backpointer = CKY(tree.leaves(), lexical_dict, syntax_dict, unary_dict)

# chart_count = [[len(tags) for tags in row] for row in chart]
# plt.imshow(chart_count)
# plt.show()

# pprint(chart[0][-1])
# visible_print(chart)
# pred_tree = build_tree(backpointer)
# pred_tree.un_chomsky_normal_form()
# pred_tree.pretty_print()
# Path("tmp/pred.txt").write_text(pred_tree.pformat(margin))
