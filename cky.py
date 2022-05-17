from nltk.tree import Tree


def fromlist(l):  # noqa
    if type(l) == list and len(l) > 0:
        label = str(l[0])
        if len(l) > 1:
            return Tree(label, [fromlist(child) for child in l[1:]])
        else:
            return label


Tree.fromlist = fromlist


def CKY(leaves: list[str], lexical_rule: dict, syntax_rule: dict, unary_rule: dict):
    n = len(leaves)
    cell: list = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i, leaf in enumerate(leaves):
        cell[i][i + 1] = [(prob, [parent, [leaf]])
                          for prob, parent in lexical_rule[leaf]]

        for prob_chain, tree in cell[i][i + 1].copy():
            tag, *_ = tree
            for prob_gen, parent in unary_rule.get(tag, []):
                prob_chain *= prob_gen
                cell[i][i + 1] += [(prob_chain, [parent, tree])]

    for l in range(2, n + 1):  # noqa
        for i in range(n - l + 1):
            j = i + l
            cand = []
            for k in range(i + 1, j):
                for prob_l, s_l in cell[i][k]:
                    tag_l, *_ = s_l
                    for prob_r, s_r in cell[k][j]:
                        tag_r, *_ = s_r
                        for prob_gen, parent in syntax_rule.get((tag_l, tag_r), []):
                            cand += [(prob_gen * prob_l * prob_r,
                                      [parent, s_l, s_r])]

            for prob_chain, tree in cell[i][j].copy():
                tag, *_ = tree
                for prob_gen, accessible in unary_rule.get(tag, []):
                    prob_chain *= prob_gen
                    cand += [(prob_chain, [accessible, tree])]

            cell[i][j] = [max(cand)] if cand else []

    return cell


def build_tree(chart, n=1):
    trees = []
    for prob, tree in chart[0][-1][:n]:
        print(prob)
        trees.append(Tree.fromlist(tree))
    return trees


def visible_print(cell):
    for row in cell[:-1]:
        print(end="|")
        for patterns in row[1:]:
            try:
                tags = ",".join([tree[0] for _, tree in patterns])
                print(f"{tags: ^10}", end="|")
            except ValueError:
                print(" " * 10, end="|")
        print()
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
        'N': [(0.3, 'NP'), (0.8, 'NNP')],
        'NP': [(0.8, 'NNP')],
    }

    chart = CKY(tree.leaves(), lexical_dict, syntax_dict, unary_dict)
    pred_tree = build_tree(chart)
    print(tree)
    print(pred_tree[0])
