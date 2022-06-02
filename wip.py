import json
from collections import defaultdict
from pprint import pprint

from nltk import Production, Tree

from src.cky import CKY
from src.counter import to_chomsky_rule, to_un_chomsky


def tmp_rule(rules: list[Production]):
    unary, syntax, lex = [defaultdict(list) for _ in range(3)]
    for rule in rules:
        left = str(rule.lhs())
        right = [str(x) for x in rule.rhs()]
        if rule.is_lexical():
            lex[right[0]].append((1, left))
            continue

        for rule2 in to_chomsky_rule(rule):
            parent, children = rule2
            if len(children) == 1:
                unary[children[0]].append((1, parent))
            else:
                # print(children)
                # print(parent)
                syntax[tuple(children)].append((1, parent))
    return unary, syntax, lex


tree: Tree = Tree.fromstring("(VP (VBZ x) (NP y) (PP z))")

unary, syntax, lex = tmp_rule(tree.productions())

tree = CKY("x y z".split(), lex, syntax, unary, beam=5)
tree.pretty_print()
tree2 = to_un_chomsky(tree)
tree2.pretty_print()
