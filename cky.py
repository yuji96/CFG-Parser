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
    cell = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    backpointer = [[{} for _ in range(n + 1)] for _ in range(n + 1)]

    for i, leaf in enumerate(leaves):
        # 単語 -> 品詞
        for prod, parent in lexical_rule[leaf]:
            cell[i][i + 1] += [(prod, parent)]
            backpointer[i][i + 1][parent] = (leaf, True)

        for prob_chain, child in cell[i][i + 1].copy():
            # unary rule（妥協）
            for _ in range(2):
                for prob_next, parent in unary_rule.get(child, []):
                    cell[i][i + 1] += [(prob_chain * prob_next, parent)]
                    backpointer[i][i + 1][parent] = (child, False)
                child = parent
                prob_chain *= prob_next

    for l in range(2, n + 1):  # noqa
        for i in range(n - l + 1):
            j = i + l
            cand = []
            for k in range(i + 1, j):
                for prob_l, s_l in cell[i][k]:
                    for prob_r, s_r in cell[k][j]:
                        for prob_gen, parent in syntax_rule.get((s_l, s_r), []):
                            cand += [(prob_gen * prob_l * prob_r, parent)]
                            backpointer[i][j][parent] = (k, s_l, s_r)

            for prob_chain, child in cand.copy():
                # unary rule（妥協）
                for _ in range(2):
                    for prob_next, parent in unary_rule.get(child, []):
                        cand += [(prob_chain * prob_next, parent)]
                        backpointer[i][j][parent] = (child, False)
                    child = parent
                    prob_chain *= prob_next

            cell[i][j] = [max(cand)] if cand else []

    return cell, backpointer


def build_tree(backpointer):
    n = len(backpointer) - 1

    k, s_l, s_r = backpointer[0][n]["S"]

    def backward(i, j, tag) -> Tree:
        unary_or_binary = backpointer[i][j][tag]
        if len(unary_or_binary) == 3:
            k, s_l, s_r = unary_or_binary
            return Tree(tag, [backward(i, k, s_l), backward(k, j, s_r)])
        else:
            child, is_terminal = unary_or_binary
            if is_terminal:
                return Tree(tag, [child])
            else:
                return Tree(tag, [backward(i, j, child)])

    return Tree("S", [backward(0, k, s_l), backward(k, n, s_r)])


def visible_print(cell):
    for row in cell[:-1]:
        print(end="|")
        for patterns in row[1:]:
            try:
                tags = ",".join([tree for _, tree in patterns])
                print(f"{tags: ^10}", end="|")
            except ValueError:
                print(" " * 10, end="|")
        print()
    print()


if __name__ == "__main__":
    from pathlib import Path

    from nltk.tree import Tree
    tree: Tree = Tree.fromstring(Path("example/tree.txt").read_text())

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
        'N': [(0.3, 'NP')],
        'NP': [(0.8, 'NNP')],
    }

    chart, backpointer = CKY(tree.leaves(), lexical_dict, syntax_dict, unary_dict)
    tree = build_tree(backpointer)
    tree.pretty_print()
