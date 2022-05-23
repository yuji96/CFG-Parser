import heapq
from heapq import nlargest

from nltk.tree import Tree


def fromlist(l):  # noqa
    if type(l) == list and len(l) > 0:
        label = str(l[0])
        if len(l) > 1:
            return Tree(label, [fromlist(child) for child in l[1:]])
        else:
            return label


Tree.fromlist = fromlist


class BackPoint:

    def __init__(self, prob, child=None, div=None, left=None, right=None,
                 is_terminal=False):
        self.prob = prob
        self.is_terminal = is_terminal
        self.child = child
        self.div = div
        self.left = left
        self.right = right
        self.is_binary = child is None

    @property
    def binary_pointers(self):
        return self.div, self.left, self.right

    @property
    def unary_pointers(self):
        return self.child, self.is_terminal

    def __lt__(self, other):
        return self.prob > other.prob


unk_tags = [(0.2286623434164418, 'NNP'), (0.17948201554758933, 'JJ'),
            (0.15265638216457889, 'NN'), (0.11841946268175776, 'CD'),
            (0.08631731582551255, 'NNS'), (0.0704877754058082, 'NNP'),
            (0.046119144479800214, 'NN'), (0.04410520803963427, 'JJ'),
            (0.03697587304144681, 'VBG'), (0.03677447939743022, 'NNP')]


def CKY(leaves: list[str], lexical_rule: dict, syntax_rule: dict, unary_rule: dict,
        beam=30):
    n = len(leaves)
    cell = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    backpointer = [[{} for _ in range(n + 1)] for _ in range(n + 1)]

    for i, leaf in enumerate(leaves):
        # 単語 -> 品詞
        for prob, parent in lexical_rule.get(leaf, unk_tags):
            cell[i][i + 1] += [(prob, parent)]
            backpointer[i][i + 1].setdefault(parent, [])
            heapq.heappush(backpointer[i][i + 1][parent],
                           BackPoint(prob, child=leaf, is_terminal=True))

        for prob_chain, child in cell[i][i + 1].copy():
            # unary rule（妥協）
            for prob_next, parent in unary_rule.get(child, []):
                prob = prob_chain * prob_next
                cell[i][i + 1] += [(prob, parent)]
                backpointer[i][i + 1].setdefault(parent, [])
                heapq.heappush(backpointer[i][i + 1][parent],
                               BackPoint(prob, child=child))

        # cell[i][i + 1] = nlargest(10, cell[i][i + 1])

    for l in range(2, n + 1):  # noqa
        for i in range(n - l + 1):
            j = i + l
            cand = []

            for k in range(i + 1, j):
                for prob_l, s_l in cell[i][k]:
                    for prob_r, s_r in cell[k][j]:
                        for prob_gen, parent in syntax_rule.get((s_l, s_r), []):
                            prob = prob_gen * prob_l * prob_r
                            cand += [(prob, parent)]
                            backpointer[i][j].setdefault(parent, [])
                            heapq.heappush(
                                backpointer[i][j][parent],
                                BackPoint(prob, div=k, left=s_l, right=s_r))

            for prob_chain, child in cand.copy():
                # unary rule（妥協）
                for prob_next, parent in unary_rule.get(child, []):
                    prob = prob_chain * prob_next
                    cand += [(prob, parent)]
                    backpointer[i][j].setdefault(parent, [])
                    heapq.heappush(backpointer[i][j][parent],
                                   BackPoint(prob, child=child))

            cell[i][j] = nlargest(beam, cand)

    return cell, backpointer


def build_tree(backpointer: list[list[dict[str, list[BackPoint]]]]) -> Tree:
    n = len(backpointer) - 1

    try:
        back_of_TOP = heapq.heappop(backpointer[0][n]["TOP"])
    except KeyError:
        return Tree("", [])

    def backward(i, j, tag) -> Tree:
        backpoint = heapq.heappop(backpointer[i][j][tag])
        if backpoint.is_binary:
            k, s_l, s_r = backpoint.binary_pointers
            return Tree(tag, [backward(i, k, s_l), backward(k, j, s_r)])
        else:
            child, is_terminal = backpoint.unary_pointers
            if is_terminal:
                return Tree(tag, [child])
            else:
                return Tree(tag, [backward(i, j, child)])

    if back_of_TOP.is_binary:
        k, s_l, s_r = back_of_TOP.binary_pointers
        return Tree("TOP", [backward(0, k, s_l), backward(k, n, s_r)])
    else:
        return Tree("TOP", [backward(0, n, back_of_TOP.child)])


def visible_print(cell):
    for row in cell[:-1]:
        print(end="|")
        for patterns in row[1:]:
            try:
                print(f"{len(patterns): ^5}", end="|")
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
