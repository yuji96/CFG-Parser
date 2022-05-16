def CKY(leaves, lexical_rule, syntax_rule, unary_rule):
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

                # [(0.7, 'N'), (0.7, 'N')]   [0.1, NNP]
                for _, child in cell[i][k]:
                    for prob, accessible, back \
                            in unary_rule.get((child, ), []):
                        cell[i][k + 1] += [(prob, accessible)]
                    # [(0.3, 'NP', ('N', )), (1.0, 'NNP', ('NP', ))],

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
        ('P', 'NP'): [(1.0, 'PP')],
    }

    unary_dict = {
        ('N', ): [(0.3, 'NP', ('N', )), (1.0, 'NNP', ('NP', ))],
        ('NP', ): [(1.0, 'NNP', ('NP', ))]
    }
    # N ->NP -> NNP
    # N -> NNP

    print(*CKY(tree.leaves(), lexical_dict, syntax_dict, unary_dict), sep="\n")


def visible_print(cell):
    for row in cell:
        for patterns in row:
            try:
                _, tags = zip(*patterns)
                tags = ",".join(set(tags))
                print(f"{tags: ^10}", end="|")
            except ValueError:
                print(" " * 10, end="|")
        print()
