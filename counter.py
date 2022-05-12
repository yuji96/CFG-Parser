from collections import Counter

from nltk.grammar import Production


def rule_as_dict(rules: list[Production]):
    lexical_all_cases: dict[str, list] = {}
    syntax_all_case: dict[str, list] = {}

    for rule in rules:
        if rule.is_lexical():
            tag = str(rule.lhs())
            word, *_ = rule.rhs()
            assert len(_) == 0
            lexical_all_cases.setdefault(tag, [])
            lexical_all_cases[tag].append(word)
        else:
            tag = str(rule.lhs())
            children = tuple(map(str, rule.rhs()))
            syntax_all_case.setdefault(tag, [])
            syntax_all_case[tag].append(children)

    lexical_dict: dict[str, list] = {}
    syntax_dict: dict[str, list] = {}
    for case_dict, out_dict in zip([lexical_all_cases, syntax_all_case],
                                   [lexical_dict, syntax_dict]):
        for tag, cases in case_dict.items():
            n = len(cases)
            for word, count in Counter(cases).items():
                out_dict.setdefault(word, [])
                out_dict[word].append((count / n, tag))

    return lexical_dict, syntax_dict


if __name__ == "__main__":
    import pickle
    from pathlib import Path as p

    from reader import read_parsed_corpus

    p("stats").mkdir(exist_ok=True)

    rules = []
    dir_numbers = range(2, 21 + 1)
    for path, tree in read_parsed_corpus("treebank_3/parsed/mrg/wsj", dir_numbers,
                                         verbose=True):
        tree.chomsky_normal_form()
        rules += tree.productions()

    lexical_dict, syntax_dict = rule_as_dict(rules)
    p("stats/lexical_dict.pkl").write_bytes(pickle.dumps(lexical_dict))
    p("stats/syntax_dict.pkl").write_bytes(pickle.dumps(syntax_dict))
