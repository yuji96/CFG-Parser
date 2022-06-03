import json
from collections import Counter, defaultdict

from nltk.grammar import Production, Nonterminal
from nltk.tree import Tree
from tqdm import tqdm


def rule_as_dict(rules: list[Production]) -> tuple[dict]:
    """構造ルールをカウントして、親子を逆転させた辞書を出力する。

    Parameters
    ----------
    rules : list[Production]
        `Tree.produces()` の返り値、またはそれを結合したもの。

    """
    # TODO: update docstrings

    lexical_all_cases = defaultdict(list)
    syntax_all_case = defaultdict(list)
    unary_all_case = defaultdict(list)

    for rule in rules:
        # なぜか "A" ではなく "'A'" が返る
        if rule.is_lexical():
            tag = str(rule.lhs())
            word, *_ = rule.rhs()
            assert len(_) == 0
            lexical_all_cases[tag].append(word)
        else:
            tag = str(rule.lhs())
            children = tuple(map(str, rule.rhs()))
            child, *_ = children
            if len(children) == 1 and child != tag:
                unary_all_case[tag].append(child)
            else:
                syntax_all_case[tag].append(children)

    lexical_dict: dict[str, list] = {}
    syntax_dict: dict[tuple[str], list] = {}
    unary_dict: dict[str, list] = {}
    # yapf: disable
    for i, (case_dict, out_dict) in enumerate(zip(
            [lexical_all_cases, syntax_all_case, unary_all_case],
            [lexical_dict, syntax_dict, unary_dict])):
        # yapf: enable
        for tag, cases in case_dict.items():
            n = len(cases)
            for children, count in Counter(cases).items():
                if i > 0 and count < 100:
                    continue

                out_dict.setdefault(children, [])
                out_dict[children].append((count / n, tag))

    return lexical_dict, syntax_dict, unary_dict


def to_chomsky_rules(rule: Production) -> list[Production]:

    root = str(rule.lhs())
    children = [str(s) for s in rule.rhs()]
    if len(children) <= 2:
        return [rule]

    head = children[0]

    def concat_info(child):
        if child is not None:
            return f"<{root}:[{head}]...{child}>"
        else:
            return f"<{root}:[{head}]>"

    rules = []
    parent = concat_info(children[-1])
    # FIXME use this Nonterminal
    rules.append(Production(root, [head]))

    for i in range(1, len(children))[::-1]:
        if i > 1:
            left_child = concat_info(children[i - 1])
        else:
            left_child = concat_info(None)
        rules.append(Production(left_child, [children[i]]))
        parent = left_child

    rules.append(Production(parent, [head]))

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

    return Tree(label, children)


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    from reader import read_cleaned_corpus

    Path("stats").mkdir(exist_ok=True)

    rules = []
    for tree in tqdm(read_cleaned_corpus("train")):
        for rule in tree.productions():
            rules.extend(to_chomsky_rules(rule))

    lexical_dict, syntax_dict, unary_dict = rule_as_dict(rules)

    Path("stats/lexical_markov2.pkl").write_bytes(pickle.dumps(lexical_dict))
    Path("stats/syntax_markov2.pkl").write_bytes(pickle.dumps(syntax_dict))
    Path("stats/unary_markov2.pkl").write_bytes(pickle.dumps(unary_dict))
