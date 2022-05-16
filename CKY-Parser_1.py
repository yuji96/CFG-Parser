def CKY(leaves, lexical_rule, syntax_rule):
    n = len(leaves)
    cell = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i, leaf in enumerate(leaves):
        cell[i][i + 1] = lexical_rule[leaf]

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l
            for k in range(i + 1, j):
                for prob_l, x_l in cell[i][k]:
                    for prob_r, x_r in cell[k][j]:
                        cell[i][j] += syntax_rule.get((x_l, x_r), [])
    return cell


if __name__ == "__main__":
    from nltk.tree import Tree
    from pathlib import Path as p
    tree: Tree = Tree.fromstring(p("example/tree.mrg").read_text())

    from pprint import pprint
    from counter import rule_as_dict

    lexical_dict = {
        'Time': [(0.7, 'N')],
        'an': [(1.0, 'D')],
        'arrow': [(0.1, 'N')],
        'flies': [(0.2, 'N'), (0.6, 'V')],
        'like': [(0.4, 'V'), (1.0, 'P')]
    }

    syntax_dict = {
        ('NP', 'VP'): [(1.0, 'S')],
        ('V', 'NP'): [(0.5, 'VP')],
        ('V', 'PP'): [(0.5, 'VP')],
        ('D', 'N'): [(0.4, 'NP')],
        ('N', 'N'): [(0.3, 'NP')],
        ('N', ): [(0.3, 'NP')],
        ('P', 'NP'): [(1.0, 'PP')],
    }

    print(*CKY(tree.leaves(), lexical_dict, syntax_dict), sep="\n")

