from collections import Counter, defaultdict

from nltk.grammar import Nonterminal, Production
from nltk.tree import Tree
from tqdm import tqdm


def count_case(rules: list[Production]) -> dict:
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

    return cases


def swap_dict(cases: dict) -> tuple[dict]:
    count_all = {
        parent: len(children_case)
        for parent, children_case in cases.items()
    }
    count_each_dict = {
        parent: Counter(children_case)
        for parent, children_case in cases.items()
    }

    lexical_dict = defaultdict(list)
    binary_dict = defaultdict(list)
    unary_dict = defaultdict(list)
    for parent, count_each in count_each_dict.items():
        N = count_all[parent]
        if N < 10:
            continue

        for (children, is_lexical), count in count_each.items():
            value = (count / N, parent)
            if is_lexical:
                lexical_dict[children[0]].append(value)
            elif len(children) == 1:
                unary_dict[children[0]].append(value)
            elif len(children) == 2:
                binary_dict[children].append(value)
            else:
                raise ValueError(f"{parent} -> {children}")

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

    Path("stats").mkdir(exist_ok=True)

    golds = read_cleaned_corpus("train")
    rules = []
    for tree in tqdm(golds):
        for rule in tree.productions():
            rules.extend(to_chomsky_rules(rule))

    cases = count_case(rules)
    # cases = pickle.loads(Path("stats/cases_norm.pkl").read_bytes())
    lexical_dict, syntax_dict, unary_dict = swap_dict(cases)
    Path("stats/cases_tmp.pkl").write_bytes(pickle.dumps(cases))
    Path("stats/lexical_tmp.pkl").write_bytes(pickle.dumps(lexical_dict))
    Path("stats/syntax_tmp.pkl").write_bytes(pickle.dumps(syntax_dict))
    Path("stats/unary_tmp.pkl").write_bytes(pickle.dumps(unary_dict))
