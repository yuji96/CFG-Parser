from collections import Counter, defaultdict
from copy import deepcopy
from itertools import chain

from nltk.grammar import Production


def rule_as_dict(rules: list[Production]):
    """構造ルールをカウントして、親子を逆転させた辞書を出力する。

    Parameters
    ----------
    rules : list[Production]
        `Tree.produces()` の返り値、またはそれを結合したもの。

    Returns
    -------
    lexical_dict : dict[str, list]
        単語を対応するタグの選択肢にマッピングする辞書。
        i.e. {'deel': [(0.7, 'N'), (0.2, 'V'), ...], ...}
    syntax_dict : dict[str, list]
        2 つの子ノードを、それらを生成できる親ノードの選択肢にマッピングする辞書。
        i.e. {('NP', 'VP'): [(1.0, 'S'), ...], ...}
    unary_dict : dict[str, list]
        再帰的に unary ルールを適用したときに到達可能な選択肢を返す辞書。
        バックポインタも返す。
        i.e. {('B',): [(1.0, 'A', ('B',))],
              ('C',): [(1.0, 'B', ('C',)), (1.0, 'A', ('B',))]}

    """
    # TODO: update docstrings

    lexical_all_cases = defaultdict(list)
    syntax_all_case = defaultdict(list)
    unary_all_case = defaultdict(list)

    for rule in rules:
        if rule.is_lexical():
            tag = str(rule.lhs())
            word, *_ = rule.rhs()
            assert len(_) == 0
            lexical_all_cases[tag].append(word)
        else:
            tag = str(rule.lhs())
            children = tuple(map(str, rule.rhs()))
            if len(children) == 1:
                unary_all_case[tag].append(children[0])
            else:
                syntax_all_case[tag].append(children)

    lexical_dict: dict[str, list] = {}
    syntax_dict: dict[tuple[str], list] = {}
    unary_dict: dict[str, list] = {}
    # yapf: disable
    for case_dict, out_dict in zip(
            [lexical_all_cases, syntax_all_case, unary_all_case],
            [lexical_dict, syntax_dict, unary_dict]):
        # yapf: enable
        for tag, cases in case_dict.items():
            n = len(cases)
            for children, count in Counter(cases).items():
                out_dict.setdefault(children, [])
                out_dict[children].append((count / n, tag))

    return lexical_dict, syntax_dict, unary_dict


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    from pprint import pprint
    from random import sample

    from reader import read_parsed_corpus

    TRAIN = True

    Path("stats").mkdir(exist_ok=True)

    rules = []
    dir_numbers = range(2, 21 + 1) if TRAIN else [0]
    for path, tree in read_parsed_corpus("treebank_3/parsed/mrg/wsj", dir_numbers,
                                         verbose=True):
        tree.chomsky_normal_form()
        rules += tree.productions()

    lexical_dict, syntax_dict, unary_dict = rule_as_dict(rules)

    if TRAIN:
        Path("stats/lexical_dict.pkl").write_bytes(pickle.dumps(lexical_dict))
        Path("stats/syntax_dict.pkl").write_bytes(pickle.dumps(syntax_dict))
        Path("stats/unary_dict.pkl").write_bytes(pickle.dumps(unary_dict))
    else:
        pprint(dict((sample(sorted(unary_dict.items()), 5))))
