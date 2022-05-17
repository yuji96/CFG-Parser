def CKY(leaves, lexical_rule, syntax_rule, unary_rule):
    n = len(leaves)
    cell = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i, leaf in enumerate(leaves):
        cell[i][i + 1] = lexical_rule[leaf]

        for _, child in cell[i][i + 1].copy():
            for prob, accessible, back \
                    in unary_rule.get((child, ), []):
                cell[i][i + 1] += [(prob, accessible)]

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l
            cand = []
            for k in range(i + 1, j):
                for prob_l, s_l in cell[i][k]:
                    for prob_r, s_r in cell[k][j]:
                        for prob_gen, a in syntax_rule.get((s_l, s_r), []):
                            cand += [(prob_gen * prob_l * prob_r, a)]

            for prob_child, child in cell[i][j].copy():
                for prob_gen, accessible, back \
                        in unary_rule.get((child, ), []):
                    cand += [(prob_child * prob_gen, accessible)]

            cell[i][j] = [max(cand)] if cand else []

    return cell


def visible_print(cell):
    for row in cell:
        for patterns in row:
            try:
                _, tags = zip(*patterns)
                tags = ",".join(tags)
                print(f"{tags: ^10}", end="|")
            except ValueError:
                print(" " * 10, end="|")
        print()


if __name__ == "__main__":
    from pathlib import Path

    from nltk.tree import Tree
    tree: Tree = Tree.fromstring(Path("example/tree.mrg").read_text())

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

    chart = CKY(tree.leaves(), lexical_dict, syntax_dict, unary_dict)

    print(chart)
    visible_print(chart)
