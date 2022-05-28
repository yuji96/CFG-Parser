from heapq import nlargest

from nltk.tree import Tree

from counter import to_un_chomsky


class MyTree:

    def __init__(self, probability, label, children: list):
        self.label = label
        self.tree = Tree(label,
                         [(child.tree if isinstance(child, MyTree) else child)
                          for child in children])
        self.prob = probability
        self.is_lexical = isinstance(children[0], str)

    def __lt__(self, other: "MyTree"):
        return self.prob < other.prob

    def __repr__(self):
        return f"{self.label} ({self.prob:.2e})"


unk_tags = [(0.2286623434164418, 'NNP'), (0.17948201554758933, 'JJ'),
            (0.15265638216457889, 'NN'), (0.11841946268175776, 'CD'),
            (0.08631731582551255, 'NNS'), (0.0704877754058082, 'NNP'),
            (0.046119144479800214, 'NN'), (0.04410520803963427, 'JJ'),
            (0.03697587304144681, 'VBG'), (0.03677447939743022, 'NNP')]


def CKY(leaves: list[str], lexical_rule: dict, syntax_rule: dict, unary_rule: dict,
        beam: int):
    n = len(leaves)
    cell: list[list[list[MyTree]]] = [[[] for _ in range(n + 1)]
                                      for _ in range(n + 1)]

    for i, leaf in enumerate(leaves):
        # 単語 -> 品詞
        for prob_gen, parent in lexical_rule.get(leaf, unk_tags):
            cell[i][i + 1].append(MyTree(prob_gen, parent, [leaf]))

        for tree in cell[i][i + 1].copy():
            # unary rule（1回だけ）
            for prob_gen, parent in unary_rule.get(tree.label, []):
                prob = tree.prob * prob_gen
                cell[i][i + 1].append(MyTree(prob, parent, [tree]))

        cell[i][i + 1] = nlargest(beam, cell[i][i + 1])

    for l in range(2, n + 1):  # noqa
        for i in range(n - l + 1):
            j = i + l

            for k in range(i + 1, j):
                for left in cell[i][k]:
                    for right in cell[k][j]:
                        for prob_gen, parent in syntax_rule.get(
                                (left.label, right.label), []):  # noqa yapf: disable
                            prob = prob_gen * left.prob * right.prob
                            cell[i][j].append(MyTree(prob, parent, [left, right]))

            cell[i][j] = nlargest(beam, cell[i][j])

            for _ in range(1):
                for tree in cell[i][j].copy():
                    for prob_gen, parent in unary_rule.get(tree.label, []):
                        prob = tree.prob * prob_gen
                        cell[i][j].append(MyTree(prob, parent, [tree]))

                cell[i][j] = nlargest(beam, cell[i][j])

    tree = max([tree for tree in cell[0][-1] if tree.label == "TOP"],
               default=MyTree(0, "", [""])).tree
    tree = to_un_chomsky(tree)
    return tree


def visible_print(cell):
    for row in cell[:-1]:
        print(end="|")
        for patterns in row[1:]:
            try:
                # print(f"{len(patterns): ^5}", end="|")
                print(f"{','.join([p.label for p in patterns]): ^5}", end="|")
            except ValueError:
                print(" " * 10, end="|")
        print()
    print()
