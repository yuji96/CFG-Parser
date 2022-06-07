from collections import Counter, defaultdict
from pprint import pprint

from nltk.grammar import Nonterminal, Production
from nltk.tree import Tree
from tqdm import tqdm


def culc_prob(rules: list[Production]) -> tuple[dict]:
    """構造ルールをカウントして、親子を逆転させた辞書を出力する。

    Parameters
    ----------
    rules : list[Production]
        `Tree.produces()` の返り値、またはそれを結合したもの。

    """
    # TODO: update docstrings

    cases = defaultdict(list)
    for rule in rules:
        parent = str(rule.lhs())
        children = tuple(str(s) for s in rule.rhs())
        cases[parent].append((children, rule.is_lexical()))

    lexical_prob = defaultdict(list)
    binary_prob = defaultdict(list)
    unary_prob = defaultdict(list)
    for parent, children_case in cases.items():
        N = len(children_case)
        counter = Counter(children_case)
        for (children, is_lexical), count in counter.items():
            if is_lexical:
                lexical_prob[parent].append((count / N, children[0]))
            elif len(children) == 1:
                unary_prob[parent].append((count / N, children[0]))
            elif len(children) == 2:
                binary_prob[parent].append((count / N, children))
            else:
                raise ValueError(f"{parent} -> {children}")

    return lexical_prob, binary_prob, unary_prob


def swap_dict(lexical_prob: dict, binary_prob: dict,
              unary_prob: dict) -> tuple[dict]:

    def _swap(d):
        swap_d = defaultdict(list)
        for parent, children_cases in d.items():
            for prob, children in children_cases:
                swap_d[children].append((prob, parent))
        return swap_d

    lexical_dict = _swap(lexical_prob)
    binary_dict = _swap(binary_prob)
    unary_dict = _swap(unary_prob)

    return lexical_dict, binary_dict, unary_dict


def to_chomsky_rules(rule: Production) -> list[Production]:

    root = str(rule.lhs())
    children = [str(s) for s in rule.rhs()]
    if len(children) <= 2:
        return [rule]

    head = children[0]

    def concat_info(child):
        if child is not None:
            return f"<{root}:{child}>"
        else:
            return f"<{root}:>"

    rules = []
    parent = concat_info(children[-1])
    rules.append(Production(Nonterminal(root), [Nonterminal(parent)]))

    for i in range(1, len(children))[::-1]:
        if i > 1:
            left_child = concat_info(children[i - 1])
        else:
            left_child = concat_info(None)
        rules.append(
            Production(Nonterminal(parent),
                       [Nonterminal(left_child),
                        Nonterminal(children[i])]))
        parent = left_child

    rules.append(Production(Nonterminal(parent), [Nonterminal(head)]))

    return rules


def to_un_chomsky(tree: Tree) -> Tree:
    child_tree = tree[0]
    if isinstance(child_tree, str):
        return tree

    label = tree.label()
    if not child_tree.label().startswith("<"):
        return Tree(label, [to_un_chomsky(x) for x in tree])

    children = []
    while len(child_tree) == 2:
        child_tree, right = child_tree
        children.append(right)
    children.append(child_tree[0])
    children = children[::-1]

    return Tree(label, [to_un_chomsky(x) for x in children])


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    from reader import read_cleaned_corpus

    pwd = Path(__file__).parent

    pwd.joinpath("../stats").mkdir(exist_ok=True)

    golds = read_cleaned_corpus("train")
    rules = []
    for tree in tqdm(golds):
        tree: Tree
        tree.chomsky_normal_form(horzMarkov=0, vertMarkov=2)
        rules.extend(tree.productions())

    lexical_prob, binary_prob, unary_prob = culc_prob(rules)
    suffix = "markov_0-2"
    pwd.joinpath(f"../stats/lexical_{suffix}.pkl").write_bytes(
        pickle.dumps(lexical_prob))
    pwd.joinpath(f"../stats/binary_{suffix}.pkl").write_bytes(
        pickle.dumps(binary_prob))
    pwd.joinpath(f"../stats/unary_{suffix}.pkl").write_bytes(
        pickle.dumps(unary_prob))
